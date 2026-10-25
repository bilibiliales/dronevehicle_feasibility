# DroneVehicle 标准 YOLO-OBB 基线代码

## 适用范围
只做 RGB-only 与 IR-only 的标准 Ultralytics OBB 基线；不含任何创新门控/融合模块。
默认数据路径为 `D:\dronevehicle_feasibility\dataset`。

## 1. 安装
在 `D:\dronevehicle_feasibility` 的虚拟环境中执行：
```powershell
python -m pip install -U ultralytics pillow
python -c "import torch,ultralytics; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'ultralytics',ultralytics.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```
如果 CUDA 显示 False，先修复 PyTorch/CUDA 环境，不要直接开始正式训练。

## 2. 转换与审计
把两个 `.py` 文件放到 `D:\dronevehicle_feasibility` 后：
```powershell
Set-Location D:\dronevehicle_feasibility
python .\prepare_dronevehicle_obb.py
```
转换默认 fail-fast。若 XML/文件配对有任何缺失，会停止并报告，不会静默丢弃。
输出在 `D:\dronevehicle_feasibility\yolo_obb_data`。图像优先硬链接，不额外复制；硬链接失败才复制。
检查 `data_audit.json`、`class_mapping.json` 以及 `rgb.yaml` / `ir.yaml`。
如果报告大量 missing key，先排查，不要立即加 `--allow-missing-modality`。

## 3. Smoke test（先跑这个）
```powershell
python .\train_yolo_obb_baselines.py --modality rgb --smoke --device 0 --batch 2
```
Smoke test 只用于确认数据管线能训练，不可把指标当正式结果。检查没有 label format / CUDA / OOM 错误后，再试 IR：
```powershell
python .\train_yolo_obb_baselines.py --modality ir --smoke --device 0 --batch 2
```

## 4. 正式训练
先单独运行，避免同时争用 16GB 显存：
```powershell
python .\train_yolo_obb_baselines.py --modality rgb --epochs 100 --imgsz 640 --batch 4 --workers 4 --device 0
python .\train_yolo_obb_baselines.py --modality ir  --epochs 100 --imgsz 640 --batch 4 --workers 4 --device 0
```
Windows 下脚本会把 dataloader workers 设为 0（更稳但可能较慢）；如要调整可修改脚本。若 OOM，把 batch 改为 2；不要同时训练两个模型。

## 5. 重要说明
- 使用 OBB 标签：每个目标输出 `class x1 y1 x2 y2 x3 y3 x4 y4`，坐标按原图宽高归一化。
- 类别编号按所有 train/val/test、RGB/IR XML 中出现的类别名统一排序，写入 `class_mapping.json`。
- 测试集仅用于最终评估；本脚本训练过程只使用 train 与 val。不要依据 test 指标调参。
- 转换程序要求四点多边形 `x1/y1 ... x4/y4`。若 XML 不是这个结构，会报错；不要改成水平框来绕过。
- `--force` 会删除整个输出目录，请确认输出目录没有需要保留的内容。
- YOLO-OBB 的标签格式和模型名称应在实际安装的 Ultralytics 版本中再核对；记录 `ultralytics` 版本和环境 JSON。
