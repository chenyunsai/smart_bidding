from library import *
#from sklearn.utils import shuffle
class dataset(Dataset):
    def __init__(self, file):
        print(file)
        self.data = pd.read_csv(file)
        #self.data = shuffle(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (is_estimated_price,is_deal_recently,is_deal,UrgentType,CardCode,U_CusLevel,U_CusGroupCode,U_Region,
         U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,Pacakage,U_QuoPackage,Eccn,U_Buyer,Status,KeyPoint,
         U_QCDesc,U_QuoGroupCode,U_QuoLevel,newclient,U_CardCode,is_satisfy_brand,is_satisfy_modle,is_satisfy_package,
         DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval,profitrate,dense_DSODay,
         dense_DemandQty,dense_PrePrice,dense_U_QuoQty,dense_U_QuoPrice,dense_latestprice_new,dense_dateinterval
        ) = torch.tensor(self.data.loc[index].tolist()[1:]).view(-1, 1)
        return (is_estimated_price,is_deal_recently,is_deal,UrgentType,CardCode,U_CusLevel,U_CusGroupCode,U_Region,
         U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,Pacakage,U_QuoPackage,Eccn,U_Buyer,Status,KeyPoint,
         U_QCDesc,U_QuoGroupCode,U_QuoLevel,newclient,U_CardCode,is_satisfy_brand,is_satisfy_modle,is_satisfy_package,
         DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval,profitrate,dense_DSODay,
         dense_DemandQty,dense_PrePrice,dense_U_QuoQty,dense_U_QuoPrice,dense_latestprice_new,dense_dateinterval
        )


def compute_label_probs(score):
    seq_relationship_score = F.softmax(score, dim=1)
    pred_label = seq_relationship_score.argmax(dim=1)
    seq_relationship_score = seq_relationship_score.detach().cpu().numpy()
    prob = seq_relationship_score[:, 1]
    pred_label = pred_label.detach().cpu().numpy()
    prob = prob.tolist()
    pred_label = pred_label.tolist()
    return prob, pred_label


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        labels = labels.long()
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.emb_estimated = nn.Embedding(
            2, 16
        )
        self.emb_is_deal = nn.Embedding(
            2, 16
        )
        self.emb_urgent = nn.Embedding(
            2, 16
        )
        self.emb_cus_code = nn.Embedding(1213,16)
        self.emb_cus_level = nn.Embedding(4, 16)
        self.emb_cus_groupcode = nn.Embedding(6, 16)
        self.emb_U_region = nn.Embedding(20, 16)
        self.emb_domain = nn.Embedding(11, 16)
        self.emb_brand = nn.Embedding(777, 16)
        self.emb_model = nn.Embedding(82078, 16)
        self.emb_pacakage = nn.Embedding(3476, 16)
        self.emb_eccn = nn.Embedding(4,16)
        self.emb_buy = nn.Embedding(72,16)
        self.emb_status = nn.Embedding(3,16)
        self.emb_key = nn.Embedding(2, 16)
        self.emb_qc = nn.Embedding(3, 16)
        self.emb_qgc = nn.Embedding(6, 16)
        self.emb_ql = nn.Embedding(7, 16)
        self.emb_nc = nn.Embedding(2, 16)
        self.u_cardcode = nn.Embedding(8556,16)
        self.emb_sb = nn.Embedding(2, 16)
        self.emb_sm = nn.Embedding(2, 16)
        self.emb_sp = nn.Embedding(2, 16)
        self.emb_dense_dso = nn.Embedding(7,16) # 0
        self.emb_dense_DemandQty = nn.Embedding(7,16) #-None 6
        self.emb_dense_U_QuoQty = nn.Embedding(7,16) # -None 6
        self.emb_dense_preprice = nn.Embedding(7,16)
        self.emb_dense_U_QuoPrice = nn.Embedding(7,16) # -None 6
        self.emb_dense_latestprice_new = nn.Embedding(7,16)
        self.emb_dense_dateinterval = nn.Embedding(7,16)
        self.main1 = nn.Sequential(
            nn.Linear(423, 256),
        )
        self.main2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # self.out1 = nn.Linear(64,2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, data):
        (is_estimated_price,is_deal_recently,is_deal,UrgentType,CardCode,U_CusLevel,U_CusGroupCode,U_Region,
         U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,Pacakage,U_QuoPackage,Eccn,U_Buyer,Status,KeyPoint,
         U_QCDesc,U_QuoGroupCode,U_QuoLevel,newclient,U_CardCode,is_satisfy_brand,is_satisfy_modle,is_satisfy_package,
         DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval,profitrate,dense_DSODay,
         dense_DemandQty,dense_PrePrice,dense_U_QuoQty,dense_U_QuoPrice,dense_latestprice_new,dense_dateinterval
        ) = data
        # embedding
        is_estimated_price = self.emb_estimated(is_estimated_price.int()).view(-1, 16)
        is_deal_recently = self.emb_is_deal(is_deal_recently.int()).view(-1, 16)
        UrgentType = self.emb_urgent(UrgentType.int()).view(-1, 16)
        CardCode = self.emb_cus_code(CardCode.int()).view(-1, 16)
        U_CusLevel = self.emb_cus_level(U_CusLevel.int()).view(-1, 16)
        U_CusGroupCode = self.emb_cus_groupcode(U_CusGroupCode.int()).view(-1, 16)
        U_Region = self.emb_U_region(U_Region.int()).view(-1, 16)
        U_DomainName = self.emb_domain(U_DomainName.int()).view(-1, 16)
        Brand = self.emb_brand(Brand.int()).view(-1, 16)
        U_QuoBrand = self.emb_brand(U_QuoBrand.int()).view(-1, 16)
        Modle = self.emb_model(Modle.int()).view(-1,16)
        U_QuoModle = self.emb_model(U_QuoModle.int()).view(-1,16)
        Pacakage = self.emb_pacakage(Pacakage.int()).view(-1,16)
        U_QuoPackage = self.emb_pacakage(U_QuoPackage.int()).view(-1,16)
        Eccn = self.emb_eccn(Eccn.int()).view(-1,16)
        U_Buyer = self.emb_buy(U_Buyer.int()).view(-1,16)
        Status = self.emb_status(Status.int()).view(-1,16)
        KeyPoint = self.emb_key(KeyPoint.int()).view(-1,16)
        U_QCDesc = self.emb_qc(U_QCDesc.int()).view(-1, 16)
        U_QuoGroupCode = self.emb_qgc(U_QuoGroupCode.int()).view(-1, 16)
        U_QuoLevel = self.emb_ql(U_QuoLevel.int()).view(-1, 16)
        newclient = self.emb_nc(newclient.int()).view(-1, 16)
        U_CardCode = self.u_cardcode(U_CardCode.int()).view(-1, 16)
        is_satisfy_brand = self.emb_sb(is_satisfy_brand.int()).view(-1, 16)
        is_satisfy_modle = self.emb_sm(is_satisfy_modle.int()).view(-1, 16)
        is_satisfy_package = self.emb_sp(is_satisfy_package.int()).view(-1, 16)
        dense_U_QuoPrice = self.emb_dense_U_QuoPrice(dense_U_QuoPrice.int()).view(-1, 16)
        dense_DSODay =self.emb_dense_dso(dense_DSODay.int()).view(-1, 16)
        dense_DemandQty = self.emb_dense_DemandQty(dense_DemandQty.int()).view(-1, 16)
        dense_U_QuoQty = self.emb_dense_U_QuoQty(dense_U_QuoQty.int()).view(-1, 16)
        dense_PrePrice = self.emb_dense_preprice(dense_PrePrice.int()).view(-1, 16)
        dense_latestprice_new = self.emb_dense_latestprice_new(dense_latestprice_new.int()).view(-1, 16)
        dense_dateinterval = self.emb_dense_dateinterval(dense_dateinterval.int()).view(-1, 16)
        continus = torch.cat([DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval], axis=1)
        input_ = torch.cat([is_estimated_price, is_deal_recently, UrgentType, CardCode,U_CusLevel, U_CusGroupCode,
                            U_Region, U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,
                            Pacakage,U_QuoPackage,Eccn,U_Buyer,
                            Status, KeyPoint, U_QCDesc, U_QuoGroupCode, U_QuoLevel, newclient,
                            U_CardCode, is_satisfy_brand,
                            is_satisfy_modle, is_satisfy_package,continus], axis=1)
        x = self.main1(input_)
        predict_y2 = self.main2(x)
        return predict_y2


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--train_video_path",
        default='v4_2.csv',
        type=str,
        help="The input video.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=256,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--num_negative", default=255, type=int, help="num of negative to use"
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    default_gpu = False
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    #####
    device = 'cpu'
    default_gpu = False
    n_gpu = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    num_train_optimization_steps = None
    data = dataset(args.train_video_path)
    train_size = int(len(data) * 0.9)
    test_size = len(data) - train_size
    train, test = torch.utils.data.random_split(data, [train_size, test_size])
    train_dataset = DataLoader(train, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = DataLoader(test, batch_size=args.train_batch_size, shuffle=True)
    num_train_optimization_steps = int(
        len(train)
        / args.train_batch_size
        / args.gradient_accumulation_steps
    ) * (args.num_train_epochs - args.start_epoch)

    model = Mymodel()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(0.9, 0.98),
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )


    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    loss1 = focal_loss(alpha=[0.06833, 0.93166])
    loss2 = nn.MSELoss(reduction="mean")
    print(device)
    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):
        model.train()
        train_probs = []
        train_true_label = []
        train_pred_label = []
        test_probs = []
        test_true_label = []
        test_pred_label = []
        test_match_loss = 0
        test_mse_loss = 0
        train_mse_loss = 0
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        for step, batch in enumerate(train_dataset):
            #batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            (is_estimated_price,is_deal_recently,is_deal,UrgentType,CardCode,U_CusLevel,U_CusGroupCode,U_Region,
         U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,Pacakage,U_QuoPackage,Eccn,U_Buyer,Status,KeyPoint,
         U_QCDesc,U_QuoGroupCode,U_QuoLevel,newclient,U_CardCode,is_satisfy_brand,is_satisfy_modle,is_satisfy_package,
         DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval,profitrate,dense_DSODay,
         dense_DemandQty,dense_PrePrice,dense_U_QuoQty,dense_U_QuoPrice,dense_latestprice_new,dense_dateinterval
        ) = batch
            #print(profitrate)
            #print(U_CardCode.max())
            #y1 = is_deal
            y2 = profitrate
            predict_y2 = model(
                batch)
            # match_loss = loss1(predict_y1,y1)
            mse_loss = loss2(predict_y2, y2)
            # loss = match_loss+mse_loss
            loss = mse_loss
            #y1 = y1.detach().cpu().numpy().tolist()
            y2 = y2.detach().cpu().numpy().tolist()
            # prob, pred = compute_label_probs(predict_y1)
            # train_probs.extend(prob)
            # train_pred_label.extend(pred)
            ##train_true_label.extend(y1)
            if n_gpu > 1:
                loss = loss.mean()
                # match_loss = match_loss.mean()
                mse_loss = mse_loss.mean()
            train_mse_loss += mse_loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (
                    step % (20 * args.gradient_accumulation_steps) == 0
                    and step != 0
            ):
                ex = ((step + 1) * args.train_batch_size / len(train_dataset.dataset)) * 100.
                # acc = accuracy_score(y1, pred)
                try:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tmatch_loss: {:.6f}\tmse_loss: {:.6f},ACC:{}'.format(
                        epochId, (step + 1) * args.train_batch_size, len(train_dataset.dataset),
                        ex, match_loss.item(), mse_loss.item(), acc))
                except:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tmse_loss: {:.6f}'.format(
                        epochId, (step + 1) * args.train_batch_size, len(train_dataset.dataset),
                        ex, mse_loss.item()))
        train_mse_loss = train_mse_loss / (step + 1)
        print("训练集的loss为{}".format(train_mse_loss))
        with open("result.txt","a") as f:
            line = "epch:{} 训练集：loss为{}".format(epochId,train_mse_loss)
            f.write(line)
            f.write('\r\n')
        # 测训练集效果
        # train_acc = accuracy_score(train_true_label, train_pred_label)
        # train_auc = roc_auc_score(train_true_label, train_probs)
        # print("*************************************")
        # print("训练集acc:{},auc:{}".format(train_acc, train_auc))
        # print("**************************************")

        # Do the evaluation
        torch.set_grad_enabled(False)
        numBatches = len(test_dataset)
        model.eval()
        for step, batch in enumerate(test_dataset):
            # image_ids = batch[-1
            #batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            (is_estimated_price,is_deal_recently,is_deal,UrgentType,CardCode,U_CusLevel,U_CusGroupCode,U_Region,
         U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,Pacakage,U_QuoPackage,Eccn,U_Buyer,Status,KeyPoint,
         U_QCDesc,U_QuoGroupCode,U_QuoLevel,newclient,U_CardCode,is_satisfy_brand,is_satisfy_modle,is_satisfy_package,
         DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval,profitrate,dense_DSODay,
         dense_DemandQty,dense_PrePrice,dense_U_QuoQty,dense_U_QuoPrice,dense_latestprice_new,dense_dateinterval
        ) = batch
            #y1 = is_deal
            y2 = profitrate
            predict_y2 = model(
                batch)
            # match_loss = loss1(predict_y1,y1)
            mse_loss = loss2(predict_y2, y2)
            # loss = match_loss+mse_loss
            loss = mse_loss
            #y1 = y1.detach().cpu().numpy().tolist()
            y2 = y2.detach().cpu().numpy().tolist()
            # prob, pred = compute_label_probs(predict_y1)
            # test_probs.extend(prob)
            # test_pred_label.extend(pred)
            # test_true_label.extend(y1)

            if n_gpu > 1:
                loss = loss.mean()
                # match_loss = match_loss.mean()
                mse_loss = mse_loss.mean()

            # test_match_loss += match_loss
            test_mse_loss += mse_loss
        # test_acc = accuracy_score(test_true_label, test_pred_label)
        # test_auc = roc_auc_score(test_true_label, test_probs)
        # test_match_loss = test_match_loss/(step+1)
        test_mse_loss = test_mse_loss / (step + 1)
        print("测试集的loss为{}".format(test_mse_loss))
        with open("result.txt","a") as f:
            line = "测试集：loss为{}".format(test_mse_loss)
            f.write(line)
            f.write('\r\n')
        # print("*************************************")
        # print("测试集match_loss:{},mse_loss:{},acc:{},auc:{}".format( test_match_loss.item(), test_mse_loss.item(),test_acc, test_auc))
        # print("**************************************")
        print('**************save****************')
        model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Onl
        output_model_file = os.path.join(
                "save", "pytorch_model_" + str(epochId) + ".bin"
            )
        torch.save(model_to_save.state_dict(), output_model_file)
        torch.set_grad_enabled(True)
    torch.save(model, 'smart.pkl')


if __name__ == "__main__":
    main()