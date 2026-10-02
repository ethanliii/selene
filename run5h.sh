#!/usr/bin/env bash
HOURS=5
END=$(( $(date +%s) + HOURS*3600 ))
FLAGS=(--permission-mode auto --permission-prompts none --effort ultracode)
first=1

while [ "$(date +%s)" -lt "$END" ]; do
  remaining=$(( (END - $(date +%s)) / 60 ))
  start=$(date +%s)
  echo "=== $(date) | run starting | ~${remaining} min left ===" >> run.log

  if [ "$first" = 1 ]; then
    claude -p "${FLAGS[@]}" \
      "Read PROMPT.md and execute it completely and autonomously. You have about ${remaining} minutes of working time. Keep PROGRESS.md updated and commit after every milestone." \
      >> run.log 2>&1
    first=0
  else
    claude -p -c "${FLAGS[@]}" \
      "Continue working autonomously. About ${remaining} minutes remain. Read PROGRESS.md and DECISIONS.md to see where you are. Finish any incomplete core milestone first; once all core milestones pass, keep improving the product (tests, demo polish, stretch features). Commit after each item. With under 20 minutes left, stop adding features: run the full test suite, run the demo end to end, fix defects, and update FINAL_REPORT.md." \
      >> run.log 2>&1
  fi

  if [ $(( $(date +%s) - start )) -lt 120 ]; then
    echo "=== $(date) | short run, backing off 10 min ===" >> run.log
    sleep 600
  else
    sleep 30
  fi
done
echo "=== $(date) | 5-hour window finished ===" >> run.log
