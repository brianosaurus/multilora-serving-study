#!/bin/sh
# Fails if the staged change removes lines from docs/CHANGELOG.md or any PLAN.md.
# Appends are fine. Override only with the author's approval: ALLOW_RECORD_DELETIONS=1.
[ "$ALLOW_RECORD_DELETIONS" = 1 ] && exit 0
bad=0
for f in $(git diff --cached --name-only --diff-filter=MD | grep -E '(^docs/CHANGELOG\.md$|(^|/)PLAN\.md$)'); do
  n=$(git diff --cached -U0 -- "$f" | grep -c '^-[^-]')
  d=$(git diff --cached --diff-filter=D --name-only -- "$f")
  if [ "$n" -gt 0 ] || [ -n "$d" ]; then echo "BLOCKED: $f loses $n line(s) (or is deleted)"; bad=1; fi
done
exit $bad
