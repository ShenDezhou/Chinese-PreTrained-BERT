import sys
from pytorch_transformers import BertModel

if __name__=="__main__":
    args = sys.argv[1:]
    model = BertModel.from_pretrained(args[0])
    toModel = BertModel.from_pretrained(args[1])

    for p1, p2 in zip(model.parameters(), toModel.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print("Models are different.")
            exit(0)
    print("Models are identical")