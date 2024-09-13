#!/bin/bash

# 运行 Python 脚本，并传入参数
python3 main.py \
 --use_lora --lora_weight 0.7 \
 --width 1024 --height 1024 \
 --local_weights \
 --lora_local_path "/data/Flux_train_20.3/output_refined_exhibition/flux-test-loraqkv-000005.safetensors"\
 --guidance 4 \
 --num_steps 50 \
 --prompt "The image presents a luxurious, elegant scene with a classical architectural aesthetic. At the center is a circular platform with a pool of clear blue water. The pool is bordered by a pristine white structure with gold accents, exuding a sense of refinement and grandeur. The structure has evenly spaced oval openings along its side, adding a touch of modernity to the otherwise classical design.The platform is positioned within a grand, open space framed by two towering marble columns. These columns have intricate golden capitals in the Corinthian style, further emphasizing the opulence of the scene. Above the columns is a large archway made of white marble, with light grey veining that enhances the luxurious feel. Through the archway, the background reveals a bright, clear sky with soft clouds, creating an inviting and serene atmosphere.The surroundings are primarily composed of white marble surfaces with subtle grey veins, accented by gold trims, adding to the polished and high-end look of the space. There are modern design elements incorporated into the setting, such as the smooth, minimalist glass railing on the right side of the platform."