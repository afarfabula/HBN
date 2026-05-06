将 stage1 预训练 checkpoint 放到这个目录下，文件名固定为：

- hbn_stage1_ckpt.pth

然后训练时加上：

- --use-pretrain

也可以用 `--pretrain-stage1-path` 指定任意绝对路径。
