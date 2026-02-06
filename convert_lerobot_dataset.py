"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path  

pkl_folder = Path(__file__).parent / Path("outputs/datasets/pick_box_pickle") 

# collect .pkl filenames in pkl_folder
pkl_files = sorted([p.name for p in pkl_folder.glob("*.pkl") if p.is_file()])



def main(push_to_hub: bool = False):

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id='pick_box_dataset',
        robot_type="panda",
        fps=50,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (4,),  # TBD 
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (4,),  # TBD 
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in pkl_files:
        raw_dataset_path = str(pkl_folder / raw_dataset_name)
        print(f"Processing raw dataset: {raw_dataset_path}")
        import pickle

        with open(raw_dataset_path, "rb") as f:
            one_episode = pickle.load(f)
            episode_len = len(one_episode['/actions'])
            for step in range(episode_len):
                dataset.add_frame(
                    
                    {
                        "image": one_episode['/observations/pixels/top'][step],
                        "wrist_image": one_episode['/observations/pixels/hand'][step],
                        "state": one_episode['/observations/agent_pos'][step],
                        "actions": one_episode['/actions'][step],
                    }, 
                    "pick up the box and put it into the container",
                )
                
            print(f"  Added episode of length {episode_len} steps.")
                
            dataset.save_episode()

    # # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["libero", "panda", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    main()
