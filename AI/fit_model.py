import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from model_loss import model_loss
from model_ebv import ConvModel

# Загрузка данных
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
with open('loss_params.pickle', 'rb') as f:
    loss_params = pickle.load(f)
import copy


# Создание экземпляра модели
model_conv1d = ConvModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit(model, data, lr, num_epochs, loss_params, loss_enbreeding, loss_overuse):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Перенос данных и модели на устройство
    model = model.to(device)
    data = data.to(device)
    existing_pairs_mask = loss_params['existing_pairs_mask'].to(device)
    avg_ebv_tensor = loss_params['avg_ebv_tensor'].to(device)
    diff_ebv_tensor = loss_params["diff_ebv_tensor"].to(device)
    num_cows = loss_params['num_cows']

    # Создание оптимизатора один раз
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())  # Сохраняем начальные веса как fallback
    loss_history = []

    # Проверка начального loss
    with torch.no_grad():
        output = model(data)
        initial_loss = model_loss(
            logits=output,
            existing_pairs_mask=existing_pairs_mask,
            avg_ebv_matrix=avg_ebv_tensor,
            diff_ebv_matrix=diff_ebv_tensor,
            num_cows=num_cows,
            loss_enbreeding=loss_enbreeding,
            loss_overuse=loss_overuse,
            info=False
        )
        print(f"Initial Loss: {initial_loss.item():.4f}")
        loss_history.append(initial_loss.item())

    # Обучение
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        output = model(data)

        loss = model_loss(
            logits=output,
            existing_pairs_mask=existing_pairs_mask,
            avg_ebv_matrix=avg_ebv_tensor,
            diff_ebv_matrix=diff_ebv_tensor,
            num_cows=num_cows,
            loss_enbreeding=loss_enbreeding,
            loss_overuse=loss_overuse,
            info=False
        )

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {current_loss:.4f}, Best Loss: {best_loss:.4f}')

    # Загружаем лучшие веса в модель
    model.load_state_dict(best_model_weights)
    model.eval()

    return model, loss_history

if __name__ == '__main__':
    # Загрузка данных
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('loss_params.pickle', 'rb') as f:
        loss_params = pickle.load(f)
    import copy


    # Создание экземпляра модели
    model_conv1d = ConvModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existing_pairs_mask = loss_params['existing_pairs_mask'].to(device)
    avg_ebv_tensor = loss_params['avg_ebv_tensor'].to(device)
    diff_ebv_tensor = loss_params["diff_ebv_tensor"].to(device)
    num_cows = loss_params['num_cows']
    data = data.to(device)
    model_conv1d = ConvModel(n=2)
    model_conv1, loss_list_conv1 = fit(model = model_conv1d,
                                       data = data, lr = 0.001,
                                       num_epochs = 20000,
                                       loss_params = loss_params,
                                       loss_enbreeding = 7,
                                       loss_overuse = 3
                                       )
    # Сохранение только весов
    torch.save(model_conv1.state_dict(), 'best_model7.pth')
    with open('loss_history7.pickle', 'wb') as f:
        pickle.dump(loss_list_conv1, f)

    model_conv1d = ConvModel(n=3)
    model_conv2, loss_list_conv2 = fit(model = model_conv1d,
                                       data = data, lr = 0.01,
                                       num_epochs = 20000,
                                       loss_params = loss_params,
                                       loss_enbreeding = 7,
                                       loss_overuse = 3
                                       )
    torch.save(model_conv2.state_dict(), 'best_model8.pth')
    with open('loss_history8.pickle', 'wb') as f:
        pickle.dump(loss_list_conv2, f)
    #
    # model_conv1d = ConvModel()
    # model_conv3, loss_list_conv3 = fit(model = model_conv1d,
    #                                    data = data, lr = 0.001,
    #                                    num_epochs = 20000,
    #                                    loss_params = loss_params,
    #                                    loss_enbreeding = 50,
    #                                    loss_overuse = 50
    #                                    )
    # torch.save(model_conv3.state_dict(), 'best_model3.pth')
    # with open('loss_history3.pickle', 'wb') as f:
    #     pickle.dump(loss_list_conv3, f)
    #
    # model_conv1d = ConvModel()
    # model_conv4, loss_list_conv4 = fit(model = model_conv1d,
    #                                    data = data, lr = 0.001,
    #                                    num_epochs = 20000,
    #                                    loss_params = loss_params,
    #                                    loss_enbreeding = 7,
    #                                    loss_overuse = 3
    #                                    )
    # torch.save(model_conv4.state_dict(), 'best_model4.pth')
    # with open('loss_history4.pickle', 'wb') as f:
    #     pickle.dump(loss_list_conv4, f)
    #
    # model_conv1d = ConvModel()
    # model_conv5, loss_list_conv5 = fit(model = model_conv1d,
    #                                    data = data, lr = 0.001,
    #                                    num_epochs = 20000,
    #                                    loss_params = loss_params,
    #                                    loss_enbreeding = 6,
    #                                    loss_overuse = 4
    #                                    )
    # torch.save(model_conv5.state_dict(), 'best_model5.pth')
    # with open('loss_history5.pickle', 'wb') as f:
    #     pickle.dump(loss_list_conv5, f)
    #
    # model_conv1d = ConvModel()
    # model_conv6, loss_list_conv6 = fit(model = model_conv1d,
    #                                    data = data, lr = 0.001,
    #                                    num_epochs = 20000,
    #                                    loss_params = loss_params,
    #                                    loss_enbreeding = 5,
    #                                    loss_overuse = 5
    #                                    )
    # torch.save(model_conv6.state_dict(), 'best_model6.pth')
    # with open('loss_history6.pickle', 'wb') as f:
    #     pickle.dump(loss_list_conv6, f)

