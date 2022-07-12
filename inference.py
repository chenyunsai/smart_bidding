from library import *
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
            nn.ReLU(),
        )
        # self.out1 = nn.Linear(64,2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, data):
        (is_estimated_price,is_deal_recently,UrgentType,CardCode,U_CusLevel,U_CusGroupCode,U_Region,
         U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,Pacakage,U_QuoPackage,Eccn,U_Buyer,Status,KeyPoint,
         U_QCDesc,U_QuoGroupCode,U_QuoLevel,newclient,U_CardCode,is_satisfy_brand,is_satisfy_modle,is_satisfy_package,
         DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval,dense_DSODay,
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
        continus = torch.cat([DSODay,DemandQty,PrePrice,U_QuoQty,U_QuoPrice,latestprice_new,dateinterval], axis=0).unsqueeze(0)
        input_ = torch.cat([is_estimated_price, is_deal_recently, UrgentType, CardCode,U_CusLevel, U_CusGroupCode,
                            U_Region, U_DomainName,Brand,U_QuoBrand,Modle,U_QuoModle,
                            Pacakage,U_QuoPackage,Eccn,U_Buyer,
                            Status, KeyPoint, U_QCDesc, U_QuoGroupCode, U_QuoLevel, newclient,
                            U_CardCode, is_satisfy_brand,
                            is_satisfy_modle, is_satisfy_package,continus], axis=1)
        x = self.main1(input_)
        predict_y2 = self.main2(x)
        return predict_y2
class Inference(nn.Module):
    def __init__(self):
        super(Inference, self).__init__()
        self.model = Mymodel()
        self.model.load_state_dict(torch.load('pytorch_model_99.bin'))
        with open("map_table.json", 'r') as load_f:
            self.load_dict = json.load(load_f)
    def helper(self,bins,x):
        for name, value in enumerate(bins):
            if x < value:
                return name
        return len(bins)
    def forward(self,data):
        (UrgentType, CardCode, U_CusLevel, U_CusGroupCode, U_Region,
         U_DomainName, Brand, U_QuoBrand, Modle, U_QuoModle, Pacakage, U_QuoPackage, Eccn, U_Buyer, Status, KeyPoint,
         U_QCDesc, U_QuoGroupCode, U_QuoLevel, newclient, U_CardCode,
         DSODay, DemandQty, PrePrice, U_QuoQty, U_QuoPrice, latestprice_new, dateinterval) = data
        # 校验
        if PrePrice == "None":
            is_estimated_price = 0
        else:
            is_estimated_price = 1
        if latestprice_new == 'None':
            is_deal_recently = 0
        else:
            is_deal_recently = 1
        if UrgentType == 'None' or UrgentType == "正常":
            UrgentType = 0
        elif UrgentType == "紧急":
            UrgentType = 1
        else:
            assert False
        CardCode = self.load_dict['CardCode'][CardCode]
        U_CusGroupCode = self.load_dict['U_CusGroupCode'][U_CusGroupCode]
        U_CusLevel = self.load_dict['U_CusLevel'][U_CusLevel]
        U_DomainName = self.load_dict['U_DomainName'][U_DomainName]
        Brand = self.load_dict['Brand'][Brand]
        U_QuoBrand = self.load_dict['Brand'][U_QuoBrand]
        Modle = self.load_dict['Modle'][Modle]
        U_QuoModle = self.load_dict['Modle'][U_QuoModle]
        U_Region = self.load_dict['U_Region'][U_Region]
        Pacakage = self.load_dict['Pacakage'][Pacakage]
        U_QuoPackage = self.load_dict['Pacakage'][U_QuoPackage]
        Eccn = self.load_dict['Eccn'][Eccn]
        U_Buyer = self.load_dict['U_Buyer'][U_Buyer]
        Status = self.load_dict['Status'][Status]
        KeyPoint = self.load_dict['KeyPoint'][KeyPoint]
        U_QCDesc = self.load_dict['U_QCDesc'][U_QCDesc]
        U_QuoGroupCode = self.load_dict['U_QuoGroupCode'][U_QuoGroupCode]
        U_QuoLevel = self.load_dict['U_QuoLevel'][U_QuoLevel]
        newclient = self.load_dict['newclient'][newclient]
        U_CardCode = self.load_dict['U_CardCode'][U_CardCode]
        if Brand == U_QuoBrand:
            is_satisfy_brand = 1
        else:
            is_satisfy_brand = 0
        if Modle == U_QuoModle:
            is_satisfy_modle = 1
        else:
            is_satisfy_modle = 0
        if Pacakage == U_QuoPackage:
            is_satisfy_package = 1
        else:
            is_satisfy_package = 0
        dense_DSODay = self.helper(self.load_dict['DSODay'],DSODay)
        dense_DemandQty = self.helper(self.load_dict['DemandQty'], DemandQty)
        dense_PrePrice = self.helper(self.load_dict['PrePrice'], PrePrice)
        dense_U_QuoQty = self.helper(self.load_dict['U_QuoQty'], U_QuoQty)
        dense_U_QuoPrice = self.helper(self.load_dict['U_QuoPrice'], U_QuoPrice)
        dense_latestprice_new = self.helper(self.load_dict['latestprice_new'], latestprice_new)
        dense_dateinterval = self.helper(self.load_dict['dateinterval'], dateinterval)
        ex = (is_estimated_price, is_deal_recently,UrgentType, CardCode, U_CusLevel, U_CusGroupCode, U_Region,
         U_DomainName, Brand, U_QuoBrand, Modle, U_QuoModle, Pacakage, U_QuoPackage, Eccn, U_Buyer, Status, KeyPoint,
         U_QCDesc, U_QuoGroupCode, U_QuoLevel, newclient, U_CardCode, is_satisfy_brand, is_satisfy_modle,
         is_satisfy_package,DSODay, DemandQty, PrePrice, U_QuoQty, U_QuoPrice, latestprice_new, dateinterval, dense_DSODay,
         dense_DemandQty, dense_PrePrice, dense_U_QuoQty, dense_U_QuoPrice, dense_latestprice_new, dense_dateinterval)
        result = self.model(torch.tensor(ex).view(-1,1))
        return round(((1+result)*U_QuoPrice).item(),2)
    
