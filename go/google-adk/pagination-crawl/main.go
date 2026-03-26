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

const appName = "agentkata_examples_pagination_crawl"

const instruction = `You are solving the AgentKata.dev pagination-crawl task.

Steps:
1. Call list_items with limit=5 and offset=0 to fetch the first page.
2. Each response contains "items" (array of objects with "id" and "value") and "pagination" (with "has_more" boolean and "next_offset" integer).
3. Track the item with the highest "value" across all pages.
4. If "has_more" is true, call list_items again with offset set to "next_offset" from the previous response.
5. Continue until "has_more" is false.
6. If the response contains an "error" object with code "TEMP_UNAVAILABLE", retry the same call with the same offset.

Reply with ONLY the "id" of the item with the highest value. No explanation, no quotes, no formatting.`

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
		TaskID:       cmp.Or(strings.TrimSpace(os.Getenv("AGENTKATA_TASK_ID")), "pagination-crawl"),
		GoogleAPIKey: googleKey,
	}, nil
}

type listItemsArgs struct {
	Limit  int `json:"limit"`
	Offset int `json:"offset"`
}

func newSolverAgent(ctx context.Context, cfg settings, client *agentkata.Client, usage *tokenTracker) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: cfg.GoogleAPIKey,
	})
	if err != nil {
		return nil, fmt.Errorf("create model: %w", err)
	}

	listTool, err := functiontool.New(functiontool.Config{
		Name:        "list_items",
		Description: "Fetch a page of items. Returns an items array and pagination metadata.",
	}, func(ctx tool.Context, args listItemsArgs) (map[string]any, error) {
		envelope, err := client.TaskAction(ctx, agentkata.TaskActionInput{
			TaskID: cfg.TaskID,
			Action: "items/list",
			Payload: map[string]any{
				"limit":  args.Limit,
				"offset": args.Offset,
			},
		})
		if err != nil {
			return nil, fmt.Errorf("list items at offset %d: %w", args.Offset, err)
		}
		return envelope.Data, nil
	})
	if err != nil {
		return nil, fmt.Errorf("create list_items tool: %w", err)
	}

	return llmagent.New(llmagent.Config{
		Name:                "pagination_crawl_solver",
		Description:         "Solves the AgentKata pagination-crawl task using tools.",
		Model:               model,
		Instruction:         instruction,
		Tools:               []tool.Tool{listTool},
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

	msg := genai.NewContentFromText("Solve the pagination-crawl task.", genai.RoleUser)

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
