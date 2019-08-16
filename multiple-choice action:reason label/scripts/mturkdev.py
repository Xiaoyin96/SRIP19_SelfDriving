# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 01:55:50 2019

@author: lihang
"""

import boto3
#s3 = boto3.resource('s3')
#for bucket in s3.buckets.all():
#    print(bucket.name)
    
aws_access_key_id = 'AKIAJ5ACI2RFRCEUNFJA'
aws_secret_access_key = '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'
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

client = boto3.client('mturk',endpoint_url=endpoint_url,region_name='us-east-1',aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,)
#print(client.get_account_balance()['AvailableBalance'])

#paginator = client.get_paginator('list_hits')
#response_iterator = paginator.paginate(
#    PaginationConfig={
#        'MaxItems': 123,
#        'PageSize': 123,
#        'StartingToken': 'string'
#    }
#)
#response = client.list_hits(
##    NextToken='next',
#    MaxResults=50
#)
#%%
f = open("question_html.txt", "r") 
question_html = f.read()
f.close()
#%%
respons5 = client.create_qualification_type(
    Name='selfdriving_test9',
    Keywords='keyword',
    Description='v1',
    QualificationTypeStatus='Active',
    RetryDelayInSeconds=12,
    Test=open('question1.xml').read(),
    AnswerKey=open('answer.xml').read(),
    TestDurationInSeconds=123,
    AutoGranted=False,
#    AutoGrantedValue=123
)

#%%

#%%
response6 = client.create_hit(
    MaxAssignments=3,
    AutoApprovalDelayInSeconds=123000,
    LifetimeInSeconds=123000,
    AssignmentDurationInSeconds=1230,
    Reward='1.23',
    Title='svcltest9',
    Keywords='driving',
    Description='test',
    Question=question_html,
    RequesterAnnotation='rastring',
    QualificationRequirements=[
        {
            'QualificationTypeId': '3P1HXW38UK4EYCCYXXVC9ZC467E5K2',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [
                35,
            ],

            'ActionsGuarded': 'PreviewAndAccept'
        },
    ],
    UniqueRequestToken='svcltest9',
    )
print ("https://workersandbox.mturk.com/mturk/preview?groupId=" + response6['HIT']['HITGroupId'])
#%%
list = []
response7 = client.list_workers_with_qualification_type(
    QualificationTypeId='3ZA6RUP9VEC9BYU3G93EQLZWNA2XQ3',
    MaxResults=100
)
for item in response7['Qualifications']:
    list.append(item)
Token = response7['NextToken']

while True:
    response7 = client.list_workers_with_qualification_type(
    QualificationTypeId='3ZA6RUP9VEC9BYU3G93EQLZWNA2XQ3',
    MaxResults=100,
    NextToken = Token
    )
    for item in response7['Qualifications']:
        list.append(item)
    Token = response7['NextToken']
    if len(response7['Qualifications']) < 100:
        break
#%%
count = 0
id_list = []
for item in list:
    if item['IntegerValue'] >= 35:
        count += 1
        print(item['WorkerId'],item['IntegerValue'],item['GrantTime'])
        id_list.append(item['WorkerId'])
print('Total worker num:',len(list))
print('Worker qualified:',count)
#%%
id_list_v3 = ['A1PYMROZ75S4FW','A1RLO9LNUJIW5S', 'A3BJ7MCE7OBXEY','A1NRVHUOTX0CNE','AKLKDS34ZAGML']
for id in id_list_v3:
    response8 = client.notify_workers(
        Subject='Invitation from svcl for another new batch of driving HIT',
        MessageText='Hi, we just launched a new batch of driving HIT. Your work in our previous HIT is of good quality. \
        We already assigned you a qualification to work on the new HIT. Please read the updated instruction before you start: \
        https://glhtest2.s3-us-west-2.amazonaws.com/ui4.html .If you have any detailed questions,feel free to send me a message. Thanks for working for us! Best,svcl',
        WorkerIds=[
                id,
                ]
        )
#%%
response = client.notify_workers(
    Subject='string',
    MessageText='string',
    WorkerIds=[
        'A2WEM6ZEQYLN0W',
    ]
)

#%%
id_list_v3 = ['A1PYMROZ75S4FW','A1RLO9LNUJIW5S', 'A3BJ7MCE7OBXEY','A1NRVHUOTX0CNE','AKLKDS34ZAGML']
for id in id_list_v3:
    response9 = client.associate_qualification_with_worker(
        QualificationTypeId='3S62YK7QBZ3VDSPMP922WEW47SSQDW',
        WorkerId=id,
        IntegerValue=100,
        SendNotification=True
    )
#%%
response7 = client.list_workers_with_qualification_type(
    QualificationTypeId='3S62YK7QBZ3VDSPMP922WEW47SSQDW',
    Status='Granted',
    MaxResults=100
)
#%%
response10 = client.get_qualification_score(
    QualificationTypeId='3S62YK7QBZ3VDSPMP922WEW47SSQDW',
    WorkerId='A3BJ7MCE7OBXEY'
)