import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


def save_model(PATH, model_, opt_, loss_):
    checkpt = {
        'model': model_.state_dict(),
        'opt': opt_.state_dict(),
        'loss': loss_.state_dict()
    }

    torch.save(checkpt, PATH)


def load_model(PATH, model_, loss_, opt_, device=torch.device('cpu')):
    chkpt = torch.load(PATH)
    model_.load_state_dict(chkpt['model'])
    model_.to(device)
    loss_.load_state_dict(chkpt['loss'])
    opt_.load_state_dict(chkpt['opt'])

    return model_, loss_, opt_


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device connected to', device)
    model = models.resnet50().to(device)
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.1)

    x = torch.randn((20, 3, 224, 224))
    y = torch.randint(high=1000, size=(20,))

    for epoch in range(5):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        L = loss(output, y)
        opt.zero_grad()
        L.backward()
        opt.step()
        print(L.item())

    print('Before saving...',loss(model(x), y).item())

    save_model('chkpt', model, opt, loss)

    model, loss, opt = load_model('chkpt', model, loss, opt, device)
    print('After loading...',loss(model(x), y).item())
