function [xiu,normal,t_out ]= train(A,m,warn,W,T,normal,xiu,t_out)
%基于ART1改进算法的训练函数

% A为输入样本，每一行代表一个样本数据,
% m代表竞争层神经元个数,
% warn为阈值,
% W为输入层到输出层的内星权值
% T为外星权值,
% normal为相似度,
% xiu为计算分成了几类，使用了哪几个神经元,
% t_out为输出

a=size(A);
n=a(2);



%调整T权值的矩阵
T_1=zeros(m+a(1),n)-1;
T_2=zeros(m+a(1),n)-1;
T_3=zeros(m+a(1),n)-1;
T_c=zeros(1,a(1));


for k=1:a(1)

    %对于未使用的节点在第一次使用时, 直接将它的外星权向量调整为输入的向量,
    %%
    s=zeros(1,m);
    %(3)寻找可以记忆A_k的神经元
    for j=1:m
        s(j)=W(j,:)*A(k,:)';
    end
    %由于s太小，进行处理
    %     s=exp(s);
    % (4)选择输入模式的最佳分类结果
    %并且如果有很多个一样大的，取位置最小的神经元
    count=zeros(1,m);
    for j=1:m
        if s(j)==max(s)
            %             count=j;
            count(j)=j;
        else
            count(j)=m+1;
        end
    end
    %     t=count;
    t=min(count);

    %对于未使用的节点在第一次使用时, 直接将它的外星权向量调整为输入的向量,
    if k==1
        for j=1:m
            for i=1:n
                T(j,i)=A(1,i);
            end
        end
    end
    %(5)判定是否接受识别结果
    normalA_k=(1-T(t,:))*(1-A(k,:)');
    normalTA_k=T(t,:)*A(k,:)';
    normal_x=normalTA_k+normalA_k;
    s_d=s;
    normal(k)=normal_x/n;
    if normal_x/n>warn%式子（*）
        %如果成立，转入步骤（7）
        xiu(t)=1;

    else
        %如果不成立，转入步骤（6）
        %取消识别结果
        %并将第t个神经元排除在下次识别范围之外
        %先查看这20个中的其它神经元能否满足式子（*）

        q=m-1;
        h=t;%h的取值范围为1：q+1
        restart=1;
        while restart==1

            %排除s_d中第h个神经元
            s_r=zeros(1,q);
            if h==1
                for i=1:q
                    s_r(i)=s_d(i+1);
                end
            elseif h==q+1
                for i=1:q
                    s_r(i)=s_d(i);
                end
            else
                %h在2：q中
                for i=1:h-1
                    s_r(i)=s_d(i);
                end
                for i=h+1:q+1
                    s_r(i-1)=s_d(i);
                end
            end
            %             s_r= (s_r-min(s_r) )/(max (s_r)-min (s_r));
            s_d=s_r;
            % (4)选择输入模式的最佳分类结果
            %并且如果有很多个一样大的，取位置最小的神经元
            count=zeros(1,q);
            for j=1:m

                if s(j)==max(s_r)
                    %                     count=j;
                    count(j)=j;
                else
                    count(j)=m+1;
                end

            end
            %计入该选出的神经元在s_d中的位置h
            for j=1:q
                if s_d(j)==max(s_r)
                    h=j;
                end
            end
            %             t=count;
            t=min(count);
            %(5)判定是否接受识别结果
            % normalA_k=sum(A(k,:));之前已经算过一遍，无需重复
            normalTA_k=T(t,:)*A(k,:)';
            normal_x=normalTA_k+normalA_k;
            %如果成立，转入步骤（7）
            if normal_x/n>warn%式子（*）
                xiu(t)=1;

                restart=0;%终止循环
            elseif q==1
                %都不满足，需要再增加一个新的神经元重新计算
                t=m+1;
                t_out_new=zeros(m+1,a(1));
                t_out_new(1:m,:)=t_out;
                t_out=t_out_new;

                W_new=zeros(m+1,n);
                W_new(1:m,:)=W;
                W=W_new;
                T_new=zeros(m+1,n);
                T_new(1:m,:)=T;
                T=T_new;
                %初始化

                for j=1:n
                    W(t,j)=1/(n+1);
                    T(t,j)=1;
                end

                xiu_new=zeros(1,m+1);
                xiu_new(1:m)=xiu;
                xiu_new(m+1)=1;
                xiu=xiu_new;
                m=m+1;
                %此时有m+1个输入结点
                restart=0;%终止循环
            else
                %该神经元不满足，转入下一个循环
                restart=1;
                q=q-1;
            end
        end

    end
    %%
    T_c(k)=t;%将每次选出来的神经元位置存入T_c中
    %接受识别结果，权值调整
    %调整第t个神经元相关的权值T
    if T_1(t,:)<0
        T_1(t,:)=T(t,:);
    elseif T_2(t,:)<0
        T_2(t,:)=T(t,:);
    elseif T_3(t,:)<0
        T_3(t,:)=T(t,:);
    else
        %调整T中t神经元对应的权值

        for j=1:a(2)
            q=T_1(t,j)+T_2(t,j)+T_3(t,j);
            if q>=2
                T(t,j)=1;
            elseif q<=1
                T(t,j)=0;
            end
            T_1(t,j)=-1;
            T_2(t,j)=-1;
            T_3(t,j)=-1;
        end

    end
    %     for i=1:n
    %         T(t,i)=T(t,i)*A(k,i);
    %
    %     end
    %调整W的权值
    for i=1:n
        %         T(t,i)=T(t,i)*A(k,i);
        W(t,i)=(T(t,i))/(0.5+sum(T(t,:)));

    end
    %%
    %T(t,:)=T(t,:).*A(k,:);
    %输出类别
    t_out(t,k)=1;

    %     %改进权值调整方法,采用2/3原则,T的权值调整方法
    %     if mod(k,3)==1
    %         T_c_1=zeros(1,size(T,1)*size(T,2));
    %
    %         for i=1:size(T,2)
    %             T_c_1((i-1)*size(T,1)+1:i*size(T,1))=T(i,:);
    %         end
    %     elseif mod(k,3)==2
    %         T_c_2=zeros(1,size(T,1)*size(T,2));
    %
    %         for i=1:size(T,2)
    %             T_c_2((i-1)*size(T,1)+1:i*size(T,1))=T(i,:);
    %         end
    %     elseif mod(k,3)==0
    %         T_c_3=zeros(1,size(T,1)*size(T,2));
    %
    %         for i=1:size(T,2)
    %             T_c_3((i-1)*size(T,1)+1:i*size(T,1))=T(i,:);
    %         end
    %         T_c_3=T;
    %         T_size(1)=min([size(T_c_3,1),size(T_c_2,1),size(T_c_1,1)]);
    %         for i=1:T_size(1)
    %             for j=1:T_size(2)
    %                 T_sum=T_c_1(i,j)+T_c_2(i,j)+T_c_3(i,j);
    %                 if T_sum>1
    %                     T(i,j)=1;
    %                 elseif T_sum<2
    %                     T(i,j)=0;
    %                 end
    %             end
    %          end

    %     end

    %改进权值调整方法,采用2/3原则
end
end