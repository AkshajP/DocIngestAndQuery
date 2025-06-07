import json
import os

import requests
from huggingface_hub import snapshot_download


def download_json(url):
    # 下载JSON文件
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.1.1':
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small_2501/*",
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    model_dir = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', allow_patterns=mineru_patterns)

    layoutreader_pattern = [
        "*.json",
        "*.safetensors",
    ]
    layoutreader_model_dir = snapshot_download('hantian/layoutreader', allow_patterns=layoutreader_pattern)

    model_dir = model_dir + '/models'
    print(f'model_dir is: {model_dir}')
    print(f'layoutreader_model_dir is: {layoutreader_model_dir}')

    json_url = 'https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json'
    config_file_name = 'magic-pdf.json'
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f'The configuration file has been configured successfully, the path is: {config_file}')

def configure_magic_pdf_settings(config_path="/root/magic-pdf.json"):
    """Configure magic-pdf.json with custom settings after initial creation"""
    import json
    import os
    
    # Default configuration overrides
    custom_config = {
        "device-mode": "cpu",  # Change to "cuda" or "mps" as needed
        "layout-config": {
            "_comment": "layoutlmv3 | doclayout_yolo",
            "model": "doclayout_yolo"
        },
        "formula-config": {
            "mfd_model": "yolo_v8_mfd",
            "mfr_model": "unimernet_small", 
            "enable": False
        },
        "table-config": {
            "_comment_model": "tablemaster | struct_eqtable (needs cuda) | rapid_table",
            "model": "tablemaster",
            "_comment_submodel": " unitable | slanet_plus ",
            "sub_model": "slanet_plus",
            "enable": True,
            "max_time": 400
        },
        "llm-aided-config": {
            "text_aided": {
                "api_key": "",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3.2:latest",
                "enable": False
            }
        }
    }
    
    # Allow environment variable overrides
    env_overrides = {
        "device-mode": os.getenv("MINERU_DEVICE_MODE", custom_config["device-mode"]),
        "layout-config": {
            **custom_config["layout-config"],
            "model": os.getenv("MINERU_LAYOUT_MODEL", custom_config["layout-config"]["model"])
        },
        "formula-config": {
            **custom_config["formula-config"],
            "enable": os.getenv("MINERU_FORMULA_ENABLE", "false").lower() == "true"
        },
        "table-config": {
            **custom_config["table-config"],
            "model": os.getenv("MINERU_TABLE_MODEL", custom_config["table-config"]["model"]),
            "enable": os.getenv("MINERU_TABLE_ENABLE", "true").lower() == "true"
        },
        "llm-aided-config": {
            "text_aided": {
                **custom_config["llm-aided-config"]["text_aided"],
                "base_url": os.getenv("OLLAMA_BASE_URL", custom_config["llm-aided-config"]["text_aided"]["base_url"]),
                "model": os.getenv("OLLAMA_MODEL", custom_config["llm-aided-config"]["text_aided"]["model"]),
                "enable": os.getenv("MINERU_LLM_ENABLE", "false").lower() == "true"
            }
        }
    }
    
    try:
        # Read existing config
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"📄 Loaded existing magic-pdf.json")
        else:
            print(f"⚠️  magic-pdf.json not found at {config_path}, creating new one")
            config = {}
        
        # Update with custom settings
        config.update(env_overrides)
        
        # Write back the updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Updated magic-pdf.json with custom configuration:")
        print(f"   Device mode: {config['device-mode']}")
        print(f"   Layout model: {config['layout-config']['model']}")
        print(f"   Table model: {config['table-config']['model']}")
        print(f"   Formula enabled: {config['formula-config']['enable']}")
        print(f"   LLM base URL: {config['llm-aided-config']['text_aided']['base_url']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to configure magic-pdf.json: {e}")
        return False