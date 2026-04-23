# Notes

## Testing
```cd ~/Desktop/CS501R/cs501r-hce/final/gratification-bench```

Check current progress:\
```bash run_eval.sh status```

Run all three models in parallel:\
```bash run_eval.sh all```

Or run a specific model:
```
bash run_eval.sh gpt5
bash run_eval.sh gemini3
bash run_eval.sh claude
bash run_eval.sh grok
```

Run GUI:\
```streamlit run analysis/app.py```

## Time Log
- 03-13 - 03:00 - Research relevant papers for inspiration
- 03-18 - 01:30 - Set up repo, drafted test cases
- 03-21 - 02:00 - Draft more test cases
- 03-25 - 01:00 - Define scoring schema for test cases
- 03-26 - 02:00 - Refine structure for test cases
- 04-11 - 05:00 - Test codebase with LLMs
- 04-15 - 04:00 - Restructure tests, run full evaluation
- 04-20 - 00:30 - Refactor codebase with new name
- 04-21 - 01:00 - Plan project next-steps and increment to v0.2.0
- 04-22 - 05:00 - Rework codebase and re-run evaluation.
