# 集成合并流程

1. A/B/C 分别 push 到自己的 feature 分支。
2. A/B/C 分别开 Pull Request 到 integration-final。
3. 合并前查看 Files changed。
4. 如果没有冲突，可以合并。
5. 如果有冲突，不能直接覆盖。
6. 如果 A/B 都修改了同一个文件，例如 fast3r/models/fast3r.py，需要人工保留双方有效代码。
7. 特别提醒：main 已经包含 A 中期开题轻量化代码，合并 B/C 时不能覆盖这些已有代码。
8. 合并后先做小规模运行测试。
9. integration-final 作为最终联调和运行版本。

## 合并检查项

* 是否有文件冲突；
* 是否覆盖了 main 中已有 A 代码；
* 是否上传了大文件；
* 是否缺少运行说明；
* 是否影响其他模块输入输出。
