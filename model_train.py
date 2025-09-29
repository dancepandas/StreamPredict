
from src.trainers import StreamModelTrainer
import config


def main(data_file:str=config.data_file, standard_scalar_file:str=config.standard_scalar_file)->None:

    trainer = StreamModelTrainer(data_file = data_file,standard_scalar_file=standard_scalar_file)

    # 准备数据
    train_loader, val_loader, test_loader = trainer.prepare_data(batch_size=config.batch_size)

    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
    )

    # 评估模型
    metrics = trainer.evaluate(test_loader)
    metrics_train = trainer.evaluate(train_loader)

    trainer.plot_training_history()
    trainer.plot_predictions(metrics)

if __name__ == '__main__':
    main()