close all;clear;clc;

%%

A=[1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 1 1 1;
    1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 1 1;
    1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 0 1 1 1;
    1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0;
    1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0];

%A=[E ;E ;E;E;F]
a=size(A);

%35个输入结点，2个模式
n=a(2);
m=10;
t_out=zeros(m,a(1));

%警戒参数
warn=0.9;
%正向权值W与反向权值T
W=rands(m,n);
T=rands(m,n);

%初始化
for i=1:m
    for j=1:n
        W(i,j)=1/(n+1);
        T(i,j)=1;
    end
end

%相似度的存储矩阵初始化
normal=zeros(1,a(1));

%输出层的识别神经元（1为识别，0为不识别
xiu=rands(m);%xiu为1x20的矩阵

%%
%训练
[xiu,normal,t_out]=train(A,m,warn,W,T,normal,xiu,t_out);

%%
%输出
xiu_c=find(xiu==1);
xiu_c=length(xiu_c);
fprintf('总共将输入的样本集分成了%d类\n',xiu_c);
fprintf('输出为：\n');
t_out'
fprintf('每个样本的相似度为:\n');
normal






%
% s=zeros(1,m);
% %寻找可以记忆A2的神经元
% for k=1:m
%     s(k)=W(k,:)*A2';
%     if s(k)==max(s)
%         count=k;
%     end
% end
%
% s_d=s;
% %如果和之前的神经元重复，继续寻找
% while xiu(count)==1
%     s_d(count)=-1;
%     for k=1:m
%         if s_d(k)==max(s_d)
%             count=k;
%         end
%     end
% end
%
% %确定找到的神经元序号count,并令其对应的输出为1
% xiu(count)=1;
% %权值调整
% W(count,:)=[0.1 0 0 0 0.4];
% T(count,:)=[1 0 0 0 1];
% xiu'
