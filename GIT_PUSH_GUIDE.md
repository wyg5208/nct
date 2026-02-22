# 🚀 NCT GitHub 仓库推送指南

## ✅ 准备工作检查清单

### 1. 确认文件已更新

**必需文件**：
- [x] `.gitignore` - 已创建（包含 Python、LaTeX、实验数据等）
- [x] `README.md` - 已更新至 v3.1.0（含论文实验结果）
- [ ] `papers/` - 论文源文件（可选，建议包含）
- [ ] `experiments/results/*.json` - 实验数据（建议保留，`.gitignore` 已配置忽略大文件）

### 2. 当前状态

```bash
# 查看当前 Git 状态
cd D:\python_projects\openclaw_demo\winclaw\src\NCT
git status
```

---

## 📋 Git Push 详细步骤

### Step 1: 添加新文件

```bash
# 进入 NCT 目录
cd D:\python_projects\openclaw_demo\winclaw\src\NCT

# 添加所有更改（包括新创建的 .gitignore 和更新的 README.md）
git add .gitignore README.md

# 或者添加所有未跟踪的文件
git add .
```

### Step 2: 提交更改

```bash
# 提交并写清晰的 commit message
git commit -m "docs: Update to v3.1.0 with experimental validation

- Add .gitignore for Python/LaTeX/experiment artifacts
- Update README with v3.1 experimental results:
  * Φ values across scales (0.126-0.329)
  * Free energy reduction: 83.0%
  * STDP latency: <2ms
  * Temporal association learning: r=0.733
  * Neuromodulation effect size: Cohen's d=1.41
- Add paper reference (NCT_arXiv.tex/pdf)
- Update project structure with experiments/ and papers/
- Include performance comparison table (v2.2 vs v3.0 vs v3.1实测)
- Add changelog section"
```

### Step 3: 推送到 GitHub

```bash
# 确保在 main 分支
git branch
# 应该显示：* main

# 推送到远程仓库
git push origin main
```

### Step 4: 验证推送

访问：https://github.com/wyg5208/nct.git

检查：
- ✅ `.gitignore` 已存在
- ✅ `README.md` 显示最新版本（v3.1.0）
- ✅ 项目结构完整
- ✅ 性能指标表格正确显示

---

## 🔧 常见问题解决

### Q1: 如果提示 "fatal: remote origin already exists"

```bash
# 查看远程仓库 URL
git remote -v

# 如果 URL 不正确，修改它
git remote set-url origin https://github.com/wyg5208/nct.git

# 重新推送
git push origin main
```

### Q2: 如果提示 "Updates were rejected because the remote contains work that you do not have"

```bash
# 先拉取远程更改
git pull origin main --rebase

# 解决可能的冲突（如果有）
# 然后再次推送
git push origin main
```

### Q3: 大文件无法推送

如果遇到大文件错误（>100MB），使用 Git LFS：

```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件类型
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.pth"

# 提交 .gitattributes
git add .gitattributes
git commit -m "chore: Configure Git LFS for large model files"

# 重新推送
git push origin main
```

---

## 📦 推荐推送的文件结构

```
nct/
├── .gitignore              ✅ 推送
├── README.md               ✅ 推送（已更新 v3.1.0）
├── pyproject.toml          ✅ 推送
├── requirements.txt        ✅ 推送
│
├── nct_modules/            ✅ 推送（核心代码）
│   └── *.py
│
├── experiments/            ⚠️ 选择性推送
│   ├── run_all_experiments.py  ✅ 推送
│   └── results/            ❌ 不推送（由.gitignore 忽略）
│       └── *.json
│
├── examples/               ✅ 推送
│   └── quickstart.py
│
├── tests/                  ✅ 推送
│   └── test_basic.py
│
├── docs/                   ✅ 推送
│   └── NCT 完整实施方案.md
│
└── papers/                 ⚠️ 可选
    └── neuroconscious_paper/
        ├── NCT_arXiv.tex   ✅ 建议推送（LaTeX 源文件）
        └── NCT_arXiv.pdf   ❌ 不推送（编译生成，较大）
```

---

## 🎯 最佳实践建议

### 1. 首次推送后维护

```bash
# 每次修改后的标准流程
git add <modified_files>
git commit -m "type: description of changes"
git pull origin main --rebase  # 先同步远程
git push origin main
```

### 2. Commit Message 规范

推荐使用以下前缀：
- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 代码格式化
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建/工具配置

示例：
```bash
git commit -m "feat: Add Phi calculator from attention flow"
git commit -m "docs: Update README with experimental results"
git commit -m "fix: Correct STDP gradient computation"
```

### 3. 发布 Release

当准备发布稳定版本时：

```bash
# 打标签
git tag -a v3.1.0 -m "NCT v3.1.0 - Experimental validation complete"

# 推送标签
git push origin v3.1.0
```

然后在 GitHub 上创建 Release：
1. 访问 https://github.com/wyg5208/nct/releases
2. 点击 "Draft a new release"
3. 选择标签 v3.1.0
4. 填写发布说明
5. 上传 PDF 等附件

---

## 📊 仓库质量提升建议

### 待添加内容（可选但推荐）

1. **License 文件**
   ```bash
   # 推荐 MIT License
   echo "MIT License - See LICENSE file" > LICENSE
   git add LICENSE
   git commit -m "docs: Add MIT license"
   ```

2. **CONTRIBUTING.md**
   ```markdown
   # 贡献指南
   如何提交 Issue 和 Pull Request
   代码规范要求
   ```

3. **CITATION.cff**（引用信息）
   ```yaml
   title: "NeuroConscious Transformer"
   authors:
     - family-names: "Your Name"
       orcid: "https://orcid.org/xxxx-xxxx-xxxx-xxxx"
   version: 3.1.0
   doi: 10.xxxx/zenodo.xxxxx
   ```

4. **Colab Notebook**
   - `examples/quickstart_colab.ipynb`
   - 方便他人快速体验

---

## 🔗 相关资源

- [GitHub Docs](https://docs.github.com/)
- [Git LFS](https://git-lfs.github.com/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)

---

## ✅ 最终检查

推送前确认：
- [ ] `.gitignore` 已创建且内容完整
- [ ] `README.md` 已更新至 v3.1.0
- [ ] 敏感信息已移除（API keys, passwords 等）
- [ ] 大文件已正确处理（LFS 或不推送）
- [ ] 代码可以正常运行
- [ ] 测试通过

推送后验证：
- [ ] GitHub 仓库显示最新更新
- [ ] README 渲染正确（无格式错误）
- [ ] 文件列表完整
- [ ] Clone 后可以正常运行

---

**🎉 恭喜！您的 NCT 代码已成功开源！**

下一步：准备 arXiv 提交材料（见 `arxiv_submission_checklist.md`）
