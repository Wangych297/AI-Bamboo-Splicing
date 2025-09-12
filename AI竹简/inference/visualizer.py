import os
from PIL import Image, ImageDraw, ImageFont

def visualize_topk_matches(query_image_path, topk_results, output_dir='visualization_results', topk=3):
    """
    可视化竹简缀合Top-K结果，将query图片与topK候选拼接输出，支持四种拼接方式。
    Args:
        query_image_path (str): 查询图片路径
        topk_results (list): predict_matches返回的结果列表（每个元素含'image_path', 'final_score', ...）
        output_dir (str): 输出目录
        topk (int): 显示前K个
    """
    os.makedirs(output_dir, exist_ok=True)
    query_img = Image.open(query_image_path).convert('RGB')
    for idx, result in enumerate(topk_results[:topk], 1):
        candidate_img = Image.open(result['image_path']).convert('RGB')
        # 统一高度和宽度
        h = max(query_img.height, candidate_img.height)
        w = max(query_img.width, candidate_img.width)
        q_img_h = query_img.resize((int(query_img.width * h / query_img.height), h))
        c_img_h = candidate_img.resize((int(candidate_img.width * h / candidate_img.height), h))
        q_img_v = query_img.resize((w, int(query_img.height * w / query_img.width)))
        c_img_v = candidate_img.resize((w, int(candidate_img.height * w / candidate_img.width)))
        # 四种拼接
        # 1. 横向（query左，候选右）
        img_lr = Image.new('RGB', (q_img_h.width + c_img_h.width, h), (255,255,255))
        img_lr.paste(q_img_h, (0,0))
        img_lr.paste(c_img_h, (q_img_h.width,0))
        # 2. 横向（query右，候选左）
        img_rl = Image.new('RGB', (q_img_h.width + c_img_h.width, h), (255,255,255))
        img_rl.paste(c_img_h, (0,0))
        img_rl.paste(q_img_h, (c_img_h.width,0))
        # 3. 纵向（query上，候选下）
        img_tb = Image.new('RGB', (w, q_img_v.height + c_img_v.height), (255,255,255))
        img_tb.paste(q_img_v, (0,0))
        img_tb.paste(c_img_v, (0, q_img_v.height))
        # 4. 纵向（query下，候选上）
        img_bt = Image.new('RGB', (w, q_img_v.height + c_img_v.height), (255,255,255))
        img_bt.paste(c_img_v, (0,0))
        img_bt.paste(q_img_v, (0, c_img_v.height))
        # 标注分数
        def draw_text(img, text):
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = None
            draw.text((10, 10), text, fill=(255,0,0), font=font)
        text = f"Top-{idx}  融合分数: {result['final_score']:.3f}"
        for mode, im in zip(['LR','RL','TB','BT'], [img_lr, img_rl, img_tb, img_bt]):
            draw_text(im, text + f"  [{mode}]")
            out_path = os.path.join(output_dir, f"top{idx}_{mode}_{os.path.basename(query_image_path)}_and_{os.path.basename(result['image_path'])}")
            im.save(out_path)
            print(f"已保存: {out_path}")

# 用法示例：
# from inference.visualizer import visualize_topk_matches
# results = engine.predict_matches(query_image, top_k=3)
# visualize_topk_matches(query_image, results, output_dir='visualization_results', topk=3)
