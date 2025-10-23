# 💾 DataNode1 Setup Guide (192.168.192.2)

**Role:** Worker Node - "Kho lưu trữ" của Hadoop Cluster

## 📋 DataNode Responsibilities
- **Data Storage:** Lưu trữ actual data blocks (64MB/128MB per block)
- **Block Management:** Create, delete, replicate blocks
- **Heartbeat:** Report status về NameNode
- **Task Execution:** Execute MapReduce tasks
- **Block Metadata:** Checksums, timestamps, block info

## 🏗️ Services Running on DataNode1
- **DataNode process** (HDFS Worker)
- **NodeManager** (YARN Worker)

---

## 🚀 Step-by-Step Setup

### Step 1: System Preparation
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Java 11
sudo apt install -y openjdk-11-jdk openjdk-11-jre

# 3. Verify Java
java -version
javac -version

# 4. Set JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Network Configuration
```bash
# Add cluster hostnames
sudo tee -a /etc/hosts << 'EOF'
192.168.192.1 namenode
192.168.192.2 datanode1  
192.168.192.3 datanode2
192.168.192.4 datanode3
EOF

# Test connectivity to NameNode (Critical!)
ping -c 3 namenode
ping -c 3 datanode1
ping -c 3 datanode2
ping -c 3 datanode3
```

### Step 3: Hadoop Installation
```bash
# 1. Download Hadoop
cd ~
wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz

# 2. Extract and setup
tar -xzf hadoop-3.3.6.tar.gz
sudo mv hadoop-3.3.6 /opt/hadoop
sudo chown -R $USER:$USER /opt/hadoop

# 3. Set Hadoop environment
echo 'export HADOOP_HOME=/opt/hadoop' >> ~/.bashrc
echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> ~/.bashrc  
echo 'export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin' >> ~/.bashrc
source ~/.bashrc

# 4. Verify installation
hadoop version
```

### Step 4: Create Data Directories
```bash
# DataNode specific directories
mkdir -p $HADOOP_HOME/data/datanode
mkdir -p $HADOOP_HOME/logs
mkdir -p $HADOOP_HOME/tmp

# Set permissions (Important for data integrity)
chmod 755 $HADOOP_HOME/data/datanode
chmod 755 $HADOOP_HOME/logs
chmod 755 $HADOOP_HOME/tmp

# Check available disk space (DataNode needs sufficient storage)
df -h $HADOOP_HOME/data/datanode
```

### Step 5: SSH Configuration
```bash
# 1. Generate SSH key for this node
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 2. Allow NameNode to SSH to this DataNode
# (This will be done from NameNode side, but prepare here)

# 3. Test SSH to NameNode (should work without password after NameNode setup)
# ssh namenode 'hostname'  # Run this after NameNode configures SSH keys
```

### Step 6: Node Configuration
```bash
# Create node identity file
cat > ~/node_config.json << 'EOF'
{
  "node_role": "datanode1",
  "zerotier_ip": "192.168.192.2", 
  "hostname": "datanode1",
  "services": ["DataNode", "NodeManager"],
  "data_directories": ["/opt/hadoop/data/datanode"],
  "resource_allocation": {
    "memory_mb": 1536,
    "vcores": 1
  }
}
EOF
```

### Step 7: Hadoop Configuration Files

#### 7.1 hadoop-env.sh
```bash
# Set JAVA_HOME in hadoop-env.sh
echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo "export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
```

#### 7.2 core-site.xml
```bash
cat > $HADOOP_HOME/etc/hadoop/core-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://namenode:9000</value>
        <description>Default filesystem URI - Points to NameNode</description>
    </property>
    
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/opt/hadoop/tmp</value>
        <description>Base for temporary directories</description>
    </property>
    
    <property>
        <name>hadoop.http.staticuser.user</name>
        <value>root</value>
    </property>
</configuration>
EOF
```

#### 7.3 hdfs-site.xml (DataNode Specific)
```bash
cat > $HADOOP_HOME/etc/hadoop/hdfs-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/opt/hadoop/data/datanode</value>
        <description>DataNode directory for storing blocks</description>
    </property>
    
    <property>
        <name>dfs.replication</name>
        <value>3</value>
        <description>Default block replication</description>
    </property>
    
    <property>
        <name>dfs.blocksize</name>
        <value>134217728</value>
        <description>Block size (128MB)</description>
    </property>
    
    <property>
        <name>dfs.datanode.http.address</name>
        <value>0.0.0.0:9864</value>
        <description>DataNode web interface</description>
    </property>
    
    <property>
        <name>dfs.datanode.address</name>
        <value>0.0.0.0:9866</value>
        <description>DataNode data transfer address</description>
    </property>
    
    <property>
        <name>dfs.datanode.ipc.address</name>
        <value>0.0.0.0:9867</value>
        <description>DataNode IPC address</description>
    </property>
    
    <property>
        <name>dfs.webhdfs.enabled</name>
        <value>true</value>
    </property>
</configuration>
EOF
```

#### 7.4 yarn-site.xml (NodeManager Configuration)
```bash
cat > $HADOOP_HOME/etc/hadoop/yarn-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>yarn.resourcemanager.hostname</name>
        <value>192.168.192.1</value>
        <description>ResourceManager host - use IP to avoid DNS issues</description>
    </property>
    
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>1536</value>
        <description>NodeManager memory (1.5GB for 2GB system)</description>
    </property>
    
    <property>
        <name>yarn.nodemanager.resource.cpu-vcores</name>
        <value>1</value>
        <description>NodeManager CPU cores</description>
    </property>
    
    <property>
        <name>yarn.scheduler.minimum-allocation-mb</name>
        <value>256</value>
    </property>
    
    <property>
        <name>yarn.scheduler.maximum-allocation-mb</name>
        <value>1536</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.vmem-check-enabled</name>
        <value>false</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.webapp.address</name>
        <value>0.0.0.0:8042</value>
        <description>NodeManager web interface</description>
    </property>
</configuration>
EOF
```

#### 7.5 mapred-site.xml
```bash
cat > $HADOOP_HOME/etc/hadoop/mapred-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
        <description>MapReduce framework</description>
    </property>
    
    <property>
        <name>mapreduce.application.classpath</name>
        <value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*:$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value>
    </property>
    
    <property>
        <name>mapreduce.map.memory.mb</name>
        <value>512</value>
        <description>Memory for map tasks</description>
    </property>
    
    <property>
        <name>mapreduce.reduce.memory.mb</name>
        <value>512</value>
        <description>Memory for reduce tasks</description>
    </property>
</configuration>
EOF
```

### Step 8: Firewall Configuration
```bash
# Open DataNode ports
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 9864/tcp   # DataNode HTTP
sudo ufw allow 9866/tcp   # DataNode data transfer
sudo ufw allow 9867/tcp   # DataNode IPC
sudo ufw allow 8042/tcp   # NodeManager Web UI

# Enable firewall
sudo ufw --force enable
sudo ufw status
```

### Step 9: Start DataNode Services
```bash
# Note: DataNode services are typically started from NameNode
# But can be started manually if needed

# Start DataNode
$HADOOP_HOME/bin/hdfs --daemon start datanode

# Start NodeManager
$HADOOP_HOME/bin/yarn --daemon start nodemanager

# Check if services started
sleep 5
jps
```

### Step 10: Verification
```bash
# 1. Check running processes
jps

# Expected output:
# - DataNode
# - NodeManager

# 2. Check DataNode status
hdfs dfsadmin -report

# 3. Check NodeManager registration
yarn node -list

# 4. Check DataNode web interface
echo "DataNode Web UI: http://datanode1:9864"
echo "NodeManager Web UI: http://datanode1:8042"

# 5. Test HDFS write capability
hdfs dfs -mkdir -p /test
hdfs dfs -put /etc/hostname /test/
hdfs dfs -ls /test/
```

## 🔧 DataNode Specific Operations

### Check Block Storage
```bash
# List blocks stored on this DataNode
ls -la $HADOOP_HOME/data/datanode/current/BP-*/current/finalized/subdir*/

# Check DataNode logs
tail -f $HADOOP_HOME/logs/hadoop-*-datanode-*.log
```

### Monitor DataNode Health
```bash
# Check disk usage
df -h $HADOOP_HOME/data/datanode

# Check DataNode metrics
curl -s http://datanode1:9864/jmx | grep -A 10 "VolumeInfo"

# Check heartbeat to NameNode
grep "Heartbeat" $HADOOP_HOME/logs/hadoop-*-datanode-*.log | tail -5
```

### Block Operations
```bash
# Check block reports
grep "BlockReport" $HADOOP_HOME/logs/hadoop-*-datanode-*.log | tail -5

# Check replication activity
grep "replication" $HADOOP_HOME/logs/hadoop-*-datanode-*.log | tail -5
```

## 🚨 Troubleshooting

### Common DataNode Issues:

1. **DataNode not connecting to NameNode:**
   ```bash
   # Check network connectivity
   telnet namenode 9000
   
   # Check DNS resolution
   nslookup namenode
   ```

2. **Insufficient disk space:**
   ```bash
   # Check disk usage
   df -h $HADOOP_HOME/data/datanode
   
   # Clean old logs if needed
   find $HADOOP_HOME/logs -name "*.log" -mtime +7 -delete
   ```

3. **Port conflicts:**
   ```bash
   # Check if ports are in use
   netstat -tlnp | grep -E "(9864|9866|9867|8042)"
   ```

### Log Files:
```bash
# DataNode logs
tail -f $HADOOP_HOME/logs/hadoop-*-datanode-*.log

# NodeManager logs  
tail -f $HADOOP_HOME/logs/yarn-*-nodemanager-*.log
```

## ✅ Success Criteria
- [x] DataNode process running
- [x] NodeManager process running
- [x] Connected to NameNode successfully
- [x] Registered with ResourceManager
- [x] Web UIs accessible
- [x] Can store and retrieve blocks
- [x] Heartbeat to NameNode working

## 📊 Performance Monitoring
```bash
# Check DataNode performance
hdfs dfsadmin -report | grep -A 20 "Name: 192.168.192.2"

# Check NodeManager resources
yarn node -status datanode1

# Monitor block operations
watch "hdfs dfsadmin -report | grep -E '(Live datanodes|blocks)'"
```

---

**🎯 DataNode1 is your primary data storage node - ensure it has sufficient disk space and network connectivity!**
