for port_num in $(seq 8097 8130); 
do
    python3 -m visdom.server -port $port_num &
done

#python3 -m visdom.server -port 8097 &
#python3 -m visdom.server -port 8098 &
#python3 -m visdom.server -port 8099 &
#python3 -m visdom.server -port 8100 &
#python3 -m visdom.server -port 8101 &
#python3 -m visdom.server -port 8102 &
#python3 -m visdom.server -port 8103 &
#python3 -m visdom.server -port 8104 &
#python3 -m visdom.server -port 8105 &
#python3 -m visdom.server -port 8106 &
#python3 -m visdom.server -port 8107 &
#python3 -m visdom.server -port 8108 &
#python3 -m visdom.server -port 8109 &
###python3 -m visdom.server -port 8110 &
