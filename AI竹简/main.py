import argparse
import os
import torch
import time
import pickle
from models.network_trainer import train_twin_network

# 新的推理功能
def run_new_inference(test_image=None, max_queries=None, model_path='twin_model.pth'):
    """运行新的推理逻辑"""
    print(f"启动竹简缀合推理引擎（新版本）")
    
    try:
        # 导入新的推理模块
        from inference.bamboo_slip_inference import BambooSlipInferenceEngine
        
        # 初始化推理引擎
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        engine = BambooSlipInferenceEngine(
            model_path=model_path,
            device=device
        )
        
        if test_image:
            # 单图推理模式
            print(f"单图推理模式: {test_image}")
            
            if not os.path.exists(test_image):
                print(f"错误 图片不存在: {test_image}")
                return None
            
            # 检查图片是否在测试集中
            if test_image not in engine.test_images:
                print(f"错误 图片不在测试集中: {test_image}")
                print(f"   测试集包含 {len(engine.test_images)} 张图片")
                return None
            
            results = engine.predict_matches(test_image, top_k=5)
            return results
        
        else:
            # 批量评估模式
            print(f"批量评估模式")
            if max_queries:
                print(f"   限制查询数量: {max_queries}")
            
            metrics = engine.batch_evaluate(max_queries=max_queries)
            return metrics
        
    except Exception as e:
        print(f"错误 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_training():
    """运行训练"""
    print("正在启动孪生网络训练...")
    train_ids, test_ids = train_twin_network()
    print(f"正确 训练完成: 训练集 {len(train_ids)} 个竹简编号, 测试集 {len(test_ids)} 个竹简编号")
    return train_ids, test_ids


def main():
    start_time = time.process_time()
    parser = argparse.ArgumentParser(description="竹简残片智能缀合系统 - 新版本")
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'predict', 'batch-eval'], 
                       required=True, 
                       help="运行模式：train(训练), predict(单图推理), batch-eval(批量评估)")
    parser.add_argument('--model', type=str, default='twin_model.pth',
                       help="模型文件路径")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                       default='cpu', help="运行设备")
    
    # 推理模式参数
    parser.add_argument('--test-image', type=str,
                       help="测试图片路径（单图推理模式必需）")
    parser.add_argument('--max-queries', type=int,
                       help="最大查询数量（批量评估模式，用于快速测试）")

    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，自动切换到CPU")
        args.device = 'cpu'
    
    if args.mode == 'train':
        print("训练模式")
        run_training()
    
    elif args.mode == 'predict':
        # 单图推理模式
        print("单图推理模式")
        
        if not args.test_image:
            print("错误 单图推理模式需要指定 --test-image 参数")
            print("示例: python main.py --mode predict --test-image yizhuihe/1/某图片.jpg")
            return
        
        results = run_new_inference(
            test_image=args.test_image,
            model_path=args.model
        )
        
        if results:
            print("正确 单图推理完成!")
        else:
            print("错误 单图推理失败!")
    
    elif args.mode == 'batch-eval':
        # 批量评估模式
        print("批量评估模式")
        
        metrics = run_new_inference(
            test_image=None,
            max_queries=args.max_queries,
            model_path=args.model
        )
        
        if metrics:
            print("正确 批量评估完成!")
        else:
            print("错误 批量评估失败!")

    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"\n程序执行时间: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    # 显示使用示例
    if len(os.sys.argv) == 1:
        print("竹简缀合智能系统 - 新版本")
        print("=" * 50)
        print("使用示例:")
        print("   训练:       python main.py --mode train")
        print("   单图推理:   python main.py --mode predict --test-image yizhuihe/1/某图片.jpg")
        print("   批量评估:   python main.py --mode batch-eval")
        print("   快速评估:   python main.py --mode batch-eval --max-queries 10")
        print()
        print("新版本特性:")
        print("   仅在yizhuihe测试集内进行推理")
        print("   5类竹简特征提取（宽度、断口、纹理、墨迹）")
        print("   孪生网络+传统特征双重分析")
        print("   Excel相邻关系作为评估标准")
        print("   完整的批量评估功能")
        print()
        print("注意:")
        print("   - 测试图片必须来自yizhuihe测试集")
        print("   - 推理结果基于Excel文档的相邻关系验证")
        
        # 显示测试集信息
        try:
            if os.path.exists('dataset_split.pkl'):
                with open('dataset_split.pkl', 'rb') as f:
                    dataset_split = pickle.load(f)
                    if 'test_groups' in dataset_split:
                        test_ids = dataset_split['test_groups']
                    elif 'test_ids' in dataset_split:
                        test_ids = dataset_split['test_ids']
                    else:
                        test_ids = []
                print(f"   - 当前测试集包含 {len(test_ids)} 个缀合组")
        except:
            pass
    else:
        main()
