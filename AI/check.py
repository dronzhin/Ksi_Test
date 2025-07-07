import torch
import pickle
from model_loss import model_loss
from model_ebv import ConvModel
from fit_model import fit


# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка loss_params
with open('loss_params.pickle', 'rb') as f:
    data_out = pickle.load(f)

existing_pairs_mask = data_out['existing_pairs_mask'].to(device)
avg_ebv_tensor = data_out['avg_ebv_tensor'].to(device)
diff_ebv_tensor = data_out["diff_ebv_tensor"].to(device)
num_cows = data_out['num_cows']

# Загрузка данных
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

data = data.to(device)

# Создание и загрузка модели
model1 = ConvModel().to(device)
model1.load_state_dict(torch.load('best_model4.pth', map_location=device))

# model_conv1, loss_list_conv1 = fit(model = model1,
#                                    data = data, lr = 0.001,
#                                    num_epochs = 200000,
#                                    loss_params = data_out,
#                                    loss_enbreeding = 7,
#                                    loss_overuse = 3
#                                    )
# # Сохранение только весов
# torch.save(model_conv1.state_dict(), 'best_model4.pth')
# with open('loss_history43.pickle', 'wb') as f:
#     pickle.dump(loss_list_conv1, f)

# Прямой проход и вычисление потерь
with torch.no_grad():
    output1 = model1(data)

loss1 = model_loss(
    logits=output1,
    existing_pairs_mask=existing_pairs_mask,
    avg_ebv_matrix=avg_ebv_tensor,
    diff_ebv_matrix=diff_ebv_tensor,
    num_cows=num_cows,
    loss_enbreeding=7,
    loss_overuse=3,
    info=True
)
print(f"Loss after loading: {loss1.item():.4f}")
index1 = torch.argmax(output1, 1)
count = 0
usage_threshold = 0.1 * num_cows
for i in range(index1.shape[0]):
    exiting = existing_pairs_mask[i, index1[i]]
    count += exiting
print(f'{count=}')