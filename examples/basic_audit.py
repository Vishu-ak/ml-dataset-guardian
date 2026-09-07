from mlguardian import DatasetAuditor
from mlguardian.reporting import render_terminal

auditor = DatasetAuditor(target="churn")
report = auditor.audit(train="examples/data/train.csv", test="examples/data/test.csv")
print(render_terminal(report))
