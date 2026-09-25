
# This script uses GitHub CLI (gh) to create the project board, milestones, labels, and issues.
# Make sure you have 'gh' installed and are authenticated.

# 1. Create Labels
gh label create "area:teacher" -f
gh label create "area:uncertainty" -f
gh label create "area:dataset" -f
gh label create "area:pii" -f
gh label create "area:student" -f
gh label create "area:evaluation" -f
gh label create "area:routing" -f
gh label create "area:documentation" -f
gh label create "area:gpu" -f

gh label create "priority:critical" --color "B60205" -f
gh label create "priority:high" --color "D93F0B" -f
gh label create "priority:medium" --color "FBCA04" -f
gh label create "priority:low" --color "0E8A16" -f

gh label create "day:1" -f
gh label create "day:2" -f
gh label create "day:3" -f

gh label create "type:task" -f
gh label create "type:experiment" -f
gh label create "type:bug" -f
gh label create "type:documentation" -f
gh label create "type:integration" -f

# 2. Create Milestones
# Note: gh issue doesn't currently support creating milestones natively without API calls.
gh api repos/:owner/:repo/milestones -f title="DAY 1 - GPU READY" -f state="open" -f description="Core code + GPU handoff ready"
gh api repos/:owner/:repo/milestones -f title="DAY 2 - INTEGRATED" -f state="open" -f description="Complete pipeline connected and experiments running"
gh api repos/:owner/:repo/milestones -f title="DAY 3 - RESULTS + DEMO" -f state="open" -f description="Experiments finalized, results documented and demo ready"
gh api repos/:owner/:repo/milestones -f title="FEATURE FREEZE" -f state="open" -f description="No new architecture unless completely broken"

# 3. Create Day 1 Critical Issues
gh issue create --title "[DAY 1][CRITICAL] Repository + Project setup" --body "Owner: Ratul`nDay: 1`nObjective: Set up initial structure." --label "priority:critical,day:1,type:task"
gh issue create --title "[DAY 1][CRITICAL] Teacher model loading" --body "Owner: Zarif`nBackup: Ratul`nDay: 1`nObjective: Load teacher model." --label "priority:critical,day:1,type:task,area:teacher"
gh issue create --title "[DAY 1][CRITICAL] Teacher inference" --body "Owner: Zarif`nBackup: Ratul`nDay: 1`nObjective: Test 1-5 sample inputs." --label "priority:critical,day:1,type:task,area:teacher"
gh issue create --title "[DAY 1][CRITICAL] Probability/logit extraction" --body "Owner: Zarif`nBackup: Ratul`nDay: 1`nObjective: Extract logic/probability information." --label "priority:critical,day:1,type:task,area:teacher"
gh issue create --title "[DAY 1][CRITICAL] Entropy calculation" --body "Owner: Zarif`nBackup: Ratul`nDay: 1`nObjective: Implement entropy calculation." --label "priority:critical,day:1,type:task,area:uncertainty"
gh issue create --title "[DAY 1][CRITICAL] Teacher disagreement" --body "Owner: Zarif`nBackup: Ratul`nDay: 1`nObjective: Implement teacher disagreement metric." --label "priority:critical,day:1,type:task,area:uncertainty"
gh issue create --title "[DAY 1][CRITICAL] GPU smoke test" --body "Owner: Zarif`nBackup: Ratul`nDay: 1`nObjective: Verify CUDA/GPU execution." --label "priority:critical,day:1,type:task,area:gpu"
gh issue create --title "[DAY 1][CRITICAL] Dataset integration" --body "Owner: Saber`nBackup: Ratul`nDay: 1`nObjective: Find/integrate existing datasets." --label "priority:critical,day:1,type:task,area:dataset"
gh issue create --title "[DAY 1][CRITICAL] Data schema definition" --body "Owner: Saber`nBackup: Ratul`nDay: 1`nObjective: Define JSON data schema." --label "priority:critical,day:1,type:task,area:dataset"

# 4. Create Day 1 High Issues
gh issue create --title "[DAY 1][HIGH] PII detection" --body "Owner: Himel`nBackup: Saber`nDay: 1`nObjective: Detect basic PII (phone, emails, etc.)." --label "priority:high,day:1,type:task,area:pii"
gh issue create --title "[DAY 1][HIGH] PII masking" --body "Owner: Himel`nBackup: Saber`nDay: 1`nObjective: Mask detected PII using synthetic examples." --label "priority:high,day:1,type:task,area:pii"
gh issue create --title "[DAY 1][HIGH] Evaluation skeleton" --body "Owner: Sadat`nBackup: Zarif`nDay: 1`nObjective: Prepare evaluation logic for metrics." --label "priority:high,day:1,type:task,area:evaluation"
gh issue create --title "[DAY 1][HIGH] Baseline structure" --body "Owner: Sadat`nBackup: Zarif`nDay: 1`nObjective: Start with simple baselines (regex/NER)." --label "priority:high,day:1,type:task,area:evaluation"

# 5. Create Day 1 Medium Issues
gh issue create --title "[DAY 1][MEDIUM] Synthetic Banglish examples" --body "Owner: Mashrafi`nBackup: Sadat`nDay: 1`nObjective: Prepare controlled ambiguous/code-mixed examples." --label "priority:medium,day:1,type:task,area:dataset"
gh issue create --title "[DAY 1][MEDIUM] Documentation" --body "Owner: Mashrafi`nBackup: Sadat`nDay: 1`nObjective: Setup documentation and figures structure." --label "priority:medium,day:1,type:documentation,area:documentation"
gh issue create --title "[DAY 1][MEDIUM] Experiment log" --body "Owner: Mashrafi`nBackup: Sadat`nDay: 1`nObjective: Create tracking log for experiments." --label "priority:medium,day:1,type:documentation,area:documentation"
gh issue create --title "[DAY 1][MEDIUM] GPU handoff documentation" --body "Owner: Zarif`nBackup: Ratul`nDay: 1`nObjective: Document how the external GPU runner should execute the code." --label "priority:medium,day:1,type:documentation,area:gpu"

Write-Host "Please use 'gh project create' to manually create the board and link the issues, or set it up via the GitHub UI."
