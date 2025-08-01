import os
import json
import logging
from lm_eval import evaluator, utils
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager, get_task_dict

# It's good practice to set up logging to see the evaluation progress
utils.setup_logging(verbosity="INFO")
eval_logger = logging.getLogger("lm-eval")

def main():
    """
    This script replaces your bash script to allow for interactive debugging.
    
    How to use:
    1. Activate your conda environment: `conda activate eeh_env`
    2. Run this script with your Python debugger. For example, in VS Code,
       you can open this file and press F5 to "Start Debugging".
    3. Place breakpoints in the lm-evaluation-harness library files,
       such as in `lm_eval/evaluator.py`. The debugger will now stop there.
    """

    # --- Configuration from your bash script ---
    model_list = [
        "bczhang/Celerity-300M-draft",
        # "bczhang/Celerity-500M-draft",
        # "bczhang/Celerity-900M-draft",
    ]

    tasks_list = [
        "arc_easy",
        # "arc_challenge",
        # "boolq",
        # "hellaswag",
    ]

    # This should have the same number of elements as tasks_list
    shots_list = [
        0,
        # 25,
        # 0,
        # 10,
    ]
    # --- End of Configuration ---

    for model_path in model_list:
        # Use a safe name for directory creation
        model_name_safe = model_path.replace("/", "__")
        
        # Define model arguments as a string
        model_args = f"pretrained={model_path},dtype=bfloat16,parallelize=True,trust_remote_code=True"
        
        # Instantiate the model
        # You can place a breakpoint here to inspect the model object
        lm = HFLM.create_from_arg_string(model_args)

        task_manager = TaskManager()

        for i, task_name in enumerate(tasks_list):
            num_fewshot = shots_list[i]
            
            eval_logger.info(f"Running task='{task_name}' for model='{model_path}' with fewshot={num_fewshot}")

            # Get the task dictionary
            task_dict = get_task_dict([task_name], num_fewshot=num_fewshot)

            # Run the evaluation
            # **You can now place breakpoints inside evaluator.py or other library files**
            breakpoint()
            results = evaluator.evaluate(
                lm=lm,
                task_dict=task_dict,
                limit=None,
                log_samples=True, # Set to False to speed up and reduce memory
            )

            # Create output directory
            output_dir = os.path.join("output", model_name_safe)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the results to a JSON file
            output_path = os.path.join(output_dir, f"{task_name}_{num_fewshot}shot.json")
            
            dumped = json.dumps(results, indent=2, default=utils.handle_non_serializable)
            
            with open(output_path, "w") as f:
                f.write(dumped)
            
            eval_logger.info(f"Results for '{task_name}' saved to {output_path}")
            # Print the main metrics
            eval_logger.info(json.dumps(results["results"][task_name], indent=2))


if __name__ == "__main__":
    main()