import os

def test_organization():
    # Get base directory relative to this test file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "implementations"))
    folders = [
        "y1992_q_learning", "y1992_reinforce", "y1994_sarsa",
        "y2015_bnn", "y2015_ddpg", "y2015_dqn", "y2015_trpo",
        "y2016_a3c", "y2017_ppo", "y2018_iclr_den", "y2018_sac",
        "y2018_td3", "y2020_cql", "y2020_iclr_apd", "y2021_lora",
        "y2023_adalora"
    ]
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        assert os.path.isdir(folder_path), f"Folder {folder} does not exist"
        
        # Check for PDF
        pdf_file = f"{folder}.pdf"
        assert os.path.isfile(os.path.join(folder_path, pdf_file)), f"PDF {pdf_file} missing in {folder}"
        
        # Check for implementation py file (usually model.py in this repo)
        # Some might have different names, let's check for any .py file
        py_files = [f for f in os.listdir(folder_path) if f.endswith(".py") and f != "__init__.py"]
        assert len(py_files) > 0, f"No implementation py file in {folder}"
        
        # Note: Config might be missing if it wasn't in configs/ originally.
        # But for the ones we moved, we should check.
        if folder in ["y1992_q_learning", "y2015_bnn", "y2020_cql", "y2021_lora", "y2023_adalora", "y2018_iclr_den", "y2020_iclr_apd"]:
            assert os.path.isfile(os.path.join(folder_path, "config.yaml")), f"Config missing in {folder}"
