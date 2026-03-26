package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"

	agentkata "github.com/agentkata/sdk-go"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

const appName = "agentkata_examples_conditional_path"

const instruction = `You are solving the AgentKata.dev conditional-path task.

Steps:
1. Call get_config to fetch the workflow configuration.
2. The config contains a "workflow" array with steps s1-s4, plus "offset" (integer) and "suffix" (string).
3. Process steps in order by type:
   - "api": Call call_api with the step's "id" and "target". Save the returned "value".
   - "lookup": Look at the step's "branch_from" to find which earlier step's value to use, then use the step's "branch_map" to map that value to a lookup key. Call call_lookup with the step's "id" and the resolved key. Save the returned "value".
   - "calculate": Evaluate the expression locally. For "int(s2)+offset", parse s2's value as integer and add the offset from config.
   - "transform": Evaluate the transform locally. For "UPPER(s1)-s3-suffix", concatenate: uppercase s1 value, a dash, s3 value, a dash, and the suffix from config.
4. If any tool returns an error with code "TEMP_UNAVAILABLE", retry the same call.

Reply with ONLY the final s4 value. No explanation, no quotes, no formatting.`

type settings struct {
	BaseURL      string
	APIToken     string
	TaskID       string
	GoogleAPIKey string
}

type summary struct {
	TaskID     string `json:"task_id"`
	Framework  string `json:"framework"`
	Answer     string `json:"answer"`
	Submission any    `json:"submission"`
}

// tokenTracker accumulates token usage across multiple LLM calls.
type tokenTracker struct {
	promptTokens     atomic.Int64
	completionTokens atomic.Int64
}

func (t *tokenTracker) callback(_ agent.CallbackContext, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
	if resp != nil && resp.UsageMetadata != nil {
		t.promptTokens.Add(int64(resp.UsageMetadata.PromptTokenCount))
		t.completionTokens.Add(int64(resp.UsageMetadata.CandidatesTokenCount))
	}
	return nil, nil
}

func (t *tokenTracker) meta(modelName string) *agentkata.RequestMeta {
	prompt := int32(t.promptTokens.Load())
	completion := int32(t.completionTokens.Load())
	return &agentkata.RequestMeta{
		Model:            &modelName,
		PromptTokens:     &prompt,
		CompletionTokens: &completion,
	}
}

func main() {
	ctx := context.Background()

	cfg, err := loadSettings()
	if err != nil {
		fail(err)
	}

	client := agentkata.NewClient(cfg.BaseURL, cfg.APIToken, nil)
	if _, err := client.RestartTask(ctx, cfg.TaskID); err != nil {
		fail(fmt.Errorf("restart task: %w", err))
	}

	var usage tokenTracker
	solver, err := newSolverAgent(ctx, cfg, client, &usage)
	if err != nil {
		fail(fmt.Errorf("build agent: %w", err))
	}

	answer, err := runSolver(ctx, solver)
	if err != nil {
		fail(fmt.Errorf("run agent: %w", err))
	}

	submit, err := client.SubmitTask(ctx, agentkata.SubmitTaskInput{
		TaskID: cfg.TaskID,
		Answer: answer,
		Meta:   usage.meta("gemini-2.5-flash"),
	})
	if err != nil {
		fail(fmt.Errorf("submit answer: %w", err))
	}

	printJSON(summary{
		TaskID:     cfg.TaskID,
		Framework:  "google-adk",
		Answer:     answer,
		Submission: submit.Data,
	})
}

func loadSettings() (settings, error) {
	baseURL := strings.TrimSpace(os.Getenv("AGENTKATA_BASE_URL"))
	apiToken := strings.TrimSpace(os.Getenv("AGENTKATA_API_TOKEN"))
	googleKey := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))

	if baseURL == "" {
		return settings{}, errors.New("AGENTKATA_BASE_URL is required")
	}
	if apiToken == "" {
		return settings{}, errors.New("AGENTKATA_API_TOKEN is required")
	}
	if googleKey == "" {
		return settings{}, errors.New("GOOGLE_API_KEY is required")
	}

	return settings{
		BaseURL:      baseURL,
		APIToken:     apiToken,
		TaskID:       cmp.Or(strings.TrimSpace(os.Getenv("AGENTKATA_TASK_ID")), "conditional-path"),
		GoogleAPIKey: googleKey,
	}, nil
}

// Tool argument types.

type getConfigArgs struct{}

type callAPIArgs struct {
	StepID string `json:"step_id"`
	Target string `json:"target"`
}

type callLookupArgs struct {
	StepID string `json:"step_id"`
	Key    string `json:"key"`
}

func newSolverAgent(ctx context.Context, cfg settings, client *agentkata.Client, usage *tokenTracker) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: cfg.GoogleAPIKey,
	})
	if err != nil {
		return nil, fmt.Errorf("create model: %w", err)
	}

	configTool, err := functiontool.New(functiontool.Config{
		Name:        "get_config",
		Description: "Fetch the workflow configuration for the conditional-path task.",
	}, func(ctx tool.Context, _ getConfigArgs) (map[string]any, error) {
		envelope, err := client.TaskAction(ctx, agentkata.TaskActionInput{
			TaskID: cfg.TaskID,
			Action: "config",
		})
		if err != nil {
			return nil, fmt.Errorf("get config: %w", err)
		}
		return envelope.Data, nil
	})
	if err != nil {
		return nil, fmt.Errorf("create get_config tool: %w", err)
	}

	apiTool, err := functiontool.New(functiontool.Config{
		Name:        "call_api",
		Description: "Execute an API step in the workflow. Returns the step value or an error to retry.",
	}, func(ctx tool.Context, args callAPIArgs) (map[string]any, error) {
		envelope, err := client.TaskAction(ctx, agentkata.TaskActionInput{
			TaskID: cfg.TaskID,
			Action: "api",
			Payload: map[string]any{
				"step_id": args.StepID,
				"target":  args.Target,
			},
		})
		if err != nil {
			return nil, fmt.Errorf("api step %s: %w", args.StepID, err)
		}
		return envelope.Data, nil
	})
	if err != nil {
		return nil, fmt.Errorf("create call_api tool: %w", err)
	}

	lookupTool, err := functiontool.New(functiontool.Config{
		Name:        "call_lookup",
		Description: "Execute a lookup step in the workflow. Returns the step value or an error to retry.",
	}, func(ctx tool.Context, args callLookupArgs) (map[string]any, error) {
		envelope, err := client.TaskAction(ctx, agentkata.TaskActionInput{
			TaskID: cfg.TaskID,
			Action: "lookup",
			Payload: map[string]any{
				"step_id": args.StepID,
				"key":     args.Key,
			},
		})
		if err != nil {
			return nil, fmt.Errorf("lookup step %s: %w", args.StepID, err)
		}
		return envelope.Data, nil
	})
	if err != nil {
		return nil, fmt.Errorf("create call_lookup tool: %w", err)
	}

	return llmagent.New(llmagent.Config{
		Name:                "conditional_path_solver",
		Description:         "Solves the AgentKata conditional-path task using tools.",
		Model:               model,
		Instruction:         instruction,
		Tools:               []tool.Tool{configTool, apiTool, lookupTool},
		AfterModelCallbacks: []llmagent.AfterModelCallback{usage.callback},
	})
}

func runSolver(ctx context.Context, solver agent.Agent) (string, error) {
	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          solver,
		SessionService: sessionService,
	})
	if err != nil {
		return "", fmt.Errorf("create runner: %w", err)
	}

	sess, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  "solver",
	})
	if err != nil {
		return "", fmt.Errorf("create session: %w", err)
	}

	msg := genai.NewContentFromText("Solve the conditional-path task.", genai.RoleUser)

	var lastText string
	for event, runErr := range r.Run(ctx, "solver", sess.Session.ID(), msg, agent.RunConfig{}) {
		if runErr != nil {
			return "", runErr
		}
		if event == nil || event.Content == nil {
			continue
		}
		if text := contentText(event.Content); text != "" {
			lastText = text
		}
	}
	if lastText != "" {
		return lastText, nil
	}

	return "", errors.New("solver returned no final response")
}

func contentText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var parts []string
	for _, part := range content.Parts {
		if part != nil && part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, ""))
}

func printJSON(v any) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fail(fmt.Errorf("marshal output: %w", err))
	}
	fmt.Println(string(data))
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
