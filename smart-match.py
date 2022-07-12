import pymssql
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from flask import Flask, jsonify
from flask import request


app = Flask(__name__)  # 创建一个服务，赋值给APP




def helper(Domain, flag):
    if '终端用户' in Domain:
        if flag:
            return 0.95
        else:
            return 0.9
    else:
        if flag:
            return 1
        else:
            return 0.95


def count_deal(data, elps=0.00001):
    total_deal = len(data[data['TrgentEntry'] == 1])
    if total_deal:
        C = Counter(data['TrgentEntry'].tolist())
        deal_ration = C[1] / (C[0] + C[1])
        data = data[data['TrgentEntry'] == 1]
        data.reset_index(drop=True, inplace=True)
        profitrate = float(data['ProfitRate'][0])
    else:
        total_deal += elps
        deal_ration = 0 + elps
        profitrate = float(min(data['ProfitRate'].astype(float).tolist()))
    return total_deal, deal_ration, profitrate


def count_profit(total_reords, total_deals, deal_rations, profitrates, indexs):
    total_reords = total_reords / np.sum(total_reords)
    total_deals = total_deals / np.sum(total_deals)
    deal_rations = deal_rations / np.sum(deal_rations)
    sum_ = total_reords * 0.2 + total_deals * 0.3 + deal_rations * 0.5 + indexs * 0.5
    sum_ = sum_ / np.sum(sum_)
    return np.sum(sum_ * profitrates)


def bid(cardcode, Model, U_CusGroupCode, U_Region, U_DomainName, price, date=None):
    conn = pymssql.connect(host='192.168.16.207', user='sa', password='25429125Lcj', database='LCJ_SAP360_NEW')
    cursor = conn.cursor()
    cursor.execute(
        'select CardCode,Modle,TrgetEntry,ProfitRate,afterQuoPrice,DemandQty,InquiryDate from ask_price where ProfitRate>=0 and Modle = %s and CardCode=%s  order by InquiryDate desc',
        (Model, cardcode))
    tmp = cursor.fetchall()
    # 有询价记录
    if len(tmp):
        data = pd.DataFrame(tmp, columns=(
        'CardCode', 'Modle', 'TrgentEntry', 'ProfitRate', 'afterQuoPrice', 'DemandQty', 'InquiryDate'))
        if any(data['TrgentEntry'].astype(str).str.contains('1').tolist()):
            data = data[data['TrgentEntry'] == 1]
            data.reset_index(drop=True, inplace=True)
            # time
            return float(data['ProfitRate'][0])
        else:
            # 历史最低利润率降低95% 时间和最低利润率
            return min(data['ProfitRate'].astype(float).tolist()) * 0.95
    cursor.execute(
        'select Modle,U_CusGroupCode,U_Region,U_DomainName,TrgetEntry,ProfitRate,InquiryDate from ask_price where ProfitRate >= 0 and Modle = %s order by InquiryDate desc',
        (Model))
    tmp = cursor.fetchall()
    data = pd.DataFrame(tmp,
                        columns=('Modle', 'U_CusGroupCode', 'U_Region', 'U_DomainName', 'TrgentEntry', 'ProfitRate',
                                 'InquiryDate'))
    if not len(data):
        return None
    # 3
    if len(data[(data['U_CusGroupCode'] == U_CusGroupCode) & (data['U_Region'] == U_Region) & (
            data['U_DomainName'] == U_DomainName)]):
        data = data[(data['U_CusGroupCode'] == U_CusGroupCode) & (data['U_Region'] == U_Region) & (
                    data['U_DomainName'] == U_DomainName)]
        data.reset_index(drop=True, inplace=True)
        if any(data['TrgentEntry'].astype(str).str.contains('1').tolist()):
            data = data[data['TrgentEntry'] == 1]
            data.reset_index(drop=True, inplace=True)
            return float(data['ProfitRate'][0]) * helper(U_CusGroupCode, 1)
        else:
            # 历史最低利润率降低95% 时间和最低利润率
            return min(data['ProfitRate'].astype(float).tolist()) * helper(U_CusGroupCode, 0)
    # 2
    user_item = ['U_CusGroupCode', 'U_Region', 'U_DomainName']
    comb = combinations(user_item, 2)
    rate = []
    total_reords = []
    total_deals = []
    deal_rations = []
    profitrates = []
    filed_importance = np.array([0.5, 0.3, 0.2])
    indexs = []
    f_ = 0
    for index, (x, y) in enumerate(comb):
        tmp = data[(data[x] == eval(x)) & (data[y] == eval(y))]
        if len(tmp):
            tmp.reset_index(drop=True, inplace=True)
            total_deal, deal_ration, profitrate = count_deal(tmp)
            if total_deal >= 1:
                f_ = 1
            indexs.append(filed_importance[index])
            total_reords.append(len(tmp))
            total_deals.append(total_deal)
            deal_rations.append(deal_ration)
            profitrates.append(profitrate)
    if indexs:
        profit = count_profit(np.array(total_reords), np.array(total_deals), np.array(deal_rations),
                              np.array(profitrates), np.array(indexs))
        return profit * helper(U_CusGroupCode, f_)
    # 3
    rate = []
    total_reords = []
    total_deals = []
    deal_rations = []
    profitrates = []
    filed_importance = np.array([0.5, 0.3, 0.2])
    indexs = []
    f_ = 0
    for index, value in enumerate(user_item):
        tmp = data[data[value] == eval(value)]
        if len(tmp):
            tmp.reset_index(drop=True, inplace=True)
            total_deal, deal_ration, profitrate = count_deal(tmp)
            if total_deal >= 1:
                f_ = 1
            indexs.append(filed_importance[index])
            total_reords.append(len(tmp))
            total_deals.append(total_deal)
            deal_rations.append(deal_ration)
            profitrates.append(profitrate)
    if indexs:
        profit = count_profit(np.array(total_reords), np.array(total_deals), np.array(deal_rations),
                              np.array(profitrates), np.array(indexs))
        return profit * helper(U_CusGroupCode, f_)
    return None

@app.route('/main', methods=['post']) 
def main():
    result = []
    data = request.json["data"]
    for i in range(len(data)):
        tmp = bid(data[i]['cardcode'], data[i]['Model'], data[i]['U_CusGroupCode'], data[i]['U_Region'],
                  data[i]['U_DomainName'], data[i]['price'])
        if not tmp:
            print('未查询到该型号，采用兜底逻辑')
            tmp = 0.1
        result.append(round((1 + tmp) * float(data[i]['price']), 2))
    headers = {"Content-Type": "application/json"}
    json_data = {"data": result}
    return json_data, 200, headers
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5582, debug=True)
