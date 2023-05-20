import sys
from pytorch_transformers import BertModel

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



if __name__=="__main__":
    args = sys.argv[1:]
    model = BertModel.from_pretrained(args[0])

    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(f'TOTAL SIZE:{total_params/1000000.}M')
    count_parameters(model)