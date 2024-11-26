#모델들 및 파일들의 함수 가져오기
import logging
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataloader import get_transform, get_datasets, split_dataset
from model import SimpleCNN
from trainer import Trainer

# hyra는 설정 파일(train.yaml)을 통해 구성 정보를 받음
@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    # 설정 저장 및 로깅
    OmegaConf.to_yaml(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger = logging.getLogger("training")
    logger.setLevel(logging.DEBUG)

    #데이터셋 준비
    transform = get_transform()

    train_dataset, test_dataset = get_datasets(transform=transform)
    train_dataset, valid_dataset = split_dataset(dataset=train_dataset, split_size=cfg.data.train_ratio)
    logger.info(f"train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)} | test_dataset: {len(test_dataset)}\n")

    # 데이터 로더 생성
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}\n")

    # 모델 및 학습 구성
    model = SimpleCNN().to(device)
    logger.info(f"model: {model}\n")

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()

    # 학습 및 검증
    trainer = Trainer(model, train_loader, valid_loader, criterion, optimizer, device, save_dir=output_dir)
    trainer.training(num_epochs=cfg.train.num_epochs, logger=logger)

    # 테스트 평가
    test_loss, test_acc = trainer.test(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 진입점(직접 실행될 때만 main() 함수를 호출하도록 함)
if __name__ == "__main__":
    main()
