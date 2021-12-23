def data_loader(x, y, batch_size):

    batch_total = x.shape[0] // batch_size
    batch_start = 0
    batch_end = batch_size
    x_data_loader, y_data_loader = [], []

    for i in range(batch_total + 1):
        if batch_end > x.shape[0]:
            batch_end = batch_start + x.shape[0] - batch_size * batch_total
        x_data_loader.append(x[batch_start:batch_end])
        y_data_loader.append(y[batch_start:batch_end])
        batch_start += batch_size
        batch_end += batch_size

    return x_data_loader, y_data_loader
