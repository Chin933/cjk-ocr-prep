# 分类型版面识别（2026-09-22）

## 已形成的能力

当前程序已经形成可复用的栏位级版面解析流程：

- `column and subcolumn segmentation`（栏与子栏切分）：根据印刷线、重复栏距和局部几何，将页面拆成物理栏位及栏内并列小栏。
- `lane-bounded text segmentation`（按栏位边界分割文字）：候选框限制在自己的栏位内，减少跨栏大框。
- `slanted ruling-line recovery`（倾斜印刷线恢复）：用局部线段估计轻微斜率，输出贴合页面的四边形框。
- `large heading / oversized name detection`（跨栏大字／大姓名识别）：大姓名和大标题可与普通栏位分开处理。
- `continuous annotation grouping`（连续附注合并）：附注内部的短空隙不再直接造成逐字切分。
- `left-right compound glyph handling`（左右结构大字处理）：将完整左右结构字与并列小字作为不同候选处理，不再单靠中间空隙判断。
- `page-type-specific decoding`（分类型解析）：个人履历、亲属履历、Exam、文章和诗文使用不同解码路径。
- `basic circle-mark exclusion`（基础圈点排除）：文章页的规则圈点可单独生成排除层，原图保持不变。

这些能力已由 50 页多样化样本清单覆盖：Biography 28 页、Exam 10 页、Paper 12 页；其中 35 页为开发样本、15 页为留出检查样本。样本清单见 `training/layout_review/corpus_50/manifest.json`。

## 当前实现

页面类型由已核对的样本清单提供；尚未训练或验证自动页面分类器。类型标签不是人工框标注。

| 类型 | 识别流程 |
|---|---|
| 个人履历 | 检测跨栏大姓名，普通栏线在该区域停止；其余区域识别主文及附注 |
| 亲属履历 | 栏内附注概率与完整字形边界，不启用姓名跨栏合并 |
| 考试／考官 | 优先使用实际残存栏线，保留官衔中的大段纵向空白；分列小字局部处理，不使用履历附注概率判断全页角色 |
| 文章 | 提取连续正文流，工作图中分离可辨认的重复圈点，栏外文字单独成块 |
| 诗文 | 正文流、标题和作者依各自起止位置成框；较大纵向空白可分隔区域 |

原始图片保持不变。主文框是局部文字流，不代表完整人物记录或跨栏归属。

## 回归

- 13 页：个人履历 1、亲属履历 5、考试 2、文章 4、诗文 1。
- 类型清单：`training/layout_review/page_types_20260918.json`。
- 结果：`training/layout_review/batches/typed_20260918/`。
- 离线对照：根目录 `Open-Typed-Review.cmd`；原100页对照与反馈独立保留。
- `archive_113_p0054` 人工标注页的48个附注均达到 IoU≥0.5；这是拟合检查，不是跨页准确率。
- 全部39项程序测试通过。13页离线图片、切页、框显示及反馈自动保存检查通过。

复现：

```powershell
& D:/Academic/miniconda/python.exe training/experiment_page_structure.py
& D:/Academic/miniconda/python.exe training/build_note_review.py --artifacts training/layout_review/batches/typed_20260918 --baseline training/layout_review/batches/notes_100 --output review_20260918_typed.html --external-rules --review-progress --typed
& D:/Academic/miniconda/python.exe -m pytest -q
```

## 尚需专项验证

1. 考官名单中的短小分列（如“宗室”）目前偏保守，有漏检；三至五列极密小字只有基础几何处理与合成测试，尚不能宣称真实档案上已解决。
2. 与文字粘连的重墨迹不能可靠分离；文章中的真实分列小字尚未启用自动角色切换。诗文标题前同轴圈点也可能进入框。
3. 少数残损栏线仍会拉长正文框。姓名仅验证了一个真实大姓名页，兄弟并列及横排姓氏尚待扩展。
4. 履历中左右结构主字与附注混淆、附注起止仍有局部错误；阅读顺序和归属没有在这一轮重新评估。
5. 应先按类型扩充并单独验收，再评估页面／区域自动分类；混合页面需区域级类型，不能强制整页只有一种处理。

人工反馈原样存于 `training/layout_review/feedback/notes_100.user-20260917.json`，不以预测结果覆盖人工真值。
