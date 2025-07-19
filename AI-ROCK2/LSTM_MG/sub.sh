#!/bin/bash
source activate ML
echo --------------------------------------------------
echo 分子生成开始
echo 标准化用于迁移学习的分子
python cleanup_smiles.py -ft datasets/1012.smi datasets/1012cleaned.smi
echo 标准化完毕, 进行迁移学习并采样
python finetune_sampling.py --epochs 10 --size 256
echo 迁移学习及采样完毕, 分子生成结束
echo --------------------------------------------------
echo 预处理开始
echo 预处理1---去除不合理及重复分子
python p1.py
echo 预处理1---完毕
echo 预处理2---2.1去除相似分子-2.2去除不符合Lipinski原则分子与包含PAINS片段分子
python p2.py
echo 预处理2---完毕
echo 预处理3---计算QED与RAscore
python p3.py
echo 预处理3---完毕
echo 预处理结束
echo --------------------------------------------------