run: build start

build:
    DOCKER_BUILDKIT=1 docker build --network=host -t sciphi . --ssh default

start:
	docker run --rm -it \
	-p 7860:7860 \
	-p 8080:8080 \
	--gpus all \
	--ipc=host --ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v /home/javier/projects/llms/llama.cpp/models:/models \
	sciphi \
	bash -c bash
# 	bash -c "/workspace/llama.cpp/build/bin/server -m /models/codellama-34b-v2/phind-codellama-34b-v2.Q5_K_M.gguf --host 0.0.0.0 -ngl 16 -c 4096 --alias llama_2 & \
# 	    PYTHONPATH=/workspace poetry run python -u /workspace/sciphi/examples/basic_data_gen/runner.py \
#             --provider_name=llama_cpp \
#             --model_name=codellama-v2-34b \
#             --num_samples=1 \
#             --batch_size=1 \
#             --output_file_name=test.jsonl \
#             --example_config=textbooks_are_all_you_need_evol \
#             --log_level=DEBUG"

