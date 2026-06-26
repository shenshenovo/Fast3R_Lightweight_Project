# GitHub 协作流程

## 分支规则

1. main 已经包含 A 中期开题阶段的轻量化代码，不是纯原版 Fast3R。
2. main 不允许直接修改。
3. A/B/C 每个人只在自己的分支提交代码。
4. A 使用 feature/A-lightweight-final。
5. B 使用 feature/B-laf-final。
6. C 使用 feature/C-eval-final。
7. 最终代码合并到 integration-final。
8. 每个人 push 后开 Pull Request。
9. Pull Request 的 base 选择 integration-final。
10. 合并前必须查看 Files changed。
11. B/C 提交代码时不能整包覆盖仓库，只提交自己改动的文件。
12. 不上传数据集、模型权重、输出结果等大文件。

## 通用 Git 命令

```bash
git clone https://github.com/shenshenovo/Fast3R_Lightweight_Project.git
cd Fast3R_Lightweight_Project
git switch 分支名
git status
git add .
git commit -m "message"
git push
```
