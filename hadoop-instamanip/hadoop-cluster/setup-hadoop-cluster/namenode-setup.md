# 🧠 NameNode Setup Guide (192.168.192.1)

**Role:** Master Node - "Bộ não" của Hadoop Cluster

## 📋 NameNode Responsibilities
- **Metadata Manager:** Files, directories, permissions
- **Block Location:** Track blocks trên DataNodes
- **Namespace:** File system namespace management
- **Replication:** Data replication decisions
- **Resource Manager:** MapReduce job resource allocation

## 🏗️ Services Running on NameNode
- **NameNode process** (HDFS Master)
- **ResourceManager** (YARN Master)
- **Secondary NameNode** (Metadata backup)

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

# Test connectivity
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
# NameNode specific directories
mkdir -p $HADOOP_HOME/data/namenode
mkdir -p $HADOOP_HOME/data/datanode  # NameNode can also be DataNode
mkdir -p $HADOOP_HOME/logs
mkdir -p $HADOOP_HOME/tmp

# Set permissions
chmod 755 $HADOOP_HOME/data/namenode
chmod 755 $HADOOP_HOME/data/datanode
```

### Step 5: SSH Passwordless Setup (Using Existing DigitalOcean Key)
```bash
# 1. Setup authorized_keys với SSH key từ DigitalOcean
# Key này đã được add sẵn khi tạo VPS, chỉ cần setup cho Hadoop cluster
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# 2. Add SSH public key cho cluster communication
cat >> /root/.ssh/authorized_keys <<'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPUbrGATG46D/yc0yARpK/PU5OYu2fZqGCQ+3qOtEbue trantuan2k4241@gmail.com
EOF

# 3. Set correct permissions
chmod 600 /root/.ssh/authorized_keys
chown -R root:root /root/.ssh

# 4. Clean any CRLF issues
sed -i 's/\r$//' /root/.ssh/authorized_keys

# 5. Generate cluster-specific SSH key for inter-node communication
ssh-keygen -t ed25519 -f /root/.ssh/hadoop_cluster -N ""

# 6. Setup cluster SSH key as default for Hadoop
cp /root/.ssh/hadoop_cluster /root/.ssh/id_rsa
cp /root/.ssh/hadoop_cluster.pub /root/.ssh/id_rsa.pub

# 7. Add cluster key to authorized_keys
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# 8. Test local SSH (should work without password)
ssh localhost 'hostname'

# Note: DataNodes sẽ được setup với same cluster key
echo "=== SSH Setup Complete ==="
echo "✅ DigitalOcean key: Active for external access"
echo "✅ Cluster key: Ready for inter-node communication"
```

### Step 6: Hadoop Configuration Files

#### 6.1 hadoop-env.sh
```bash
# Set JAVA_HOME in hadoop-env.sh
echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo "export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# Add required user variables for Hadoop 3.x (CRITICAL FIX!)
echo 'export HDFS_NAMENODE_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export HDFS_DATANODE_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export HDFS_SECONDARYNAMENODE_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export YARN_RESOURCEMANAGER_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export YARN_NODEMANAGER_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# Verify variables added
tail -8 $HADOOP_HOME/etc/hadoop/hadoop-env.sh
```

#### 6.2 core-site.xml
```bash
cat > $HADOOP_HOME/etc/hadoop/core-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://namenode:9000</value>
        <description>Default filesystem URI</description>
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

#### 6.3 hdfs-site.xml
```bash
cat > $HADOOP_HOME/etc/hadoop/hdfs-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/opt/hadoop/data/namenode</value>
        <description>NameNode directory for namespace and transaction logs</description>
    </property>
    
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/opt/hadoop/data/datanode</value>
        <description>DataNode directory</description>
    </property>
    
    <property>
        <name>dfs.replication</name>
        <value>3</value>
        <description>Default block replication</description>
    </property>
    
    <property>
        <name>dfs.namenode.http-address</name>
        <value>0.0.0.0:9870</value>
        <description>NameNode web interface - accessible from all interfaces</description>
    </property>
    
    <property>
        <name>dfs.namenode.secondary.http-address</name>
        <value>namenode:9868</value>
        <description>Secondary NameNode web interface</description>
    </property>
    
    <property>
        <name>dfs.webhdfs.enabled</name>
        <value>true</value>
    </property>
</configuration>
EOF
```

#### 6.4 yarn-site.xml
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
        <name>yarn.resourcemanager.webapp.address</name>
        <value>0.0.0.0:8088</value>
        <description>ResourceManager web interface - accessible from all interfaces</description>
    </property>
    
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>3072</value>
        <description>NodeManager memory (3GB for 4GB system)</description>
    </property>
    
    <property>
        <name>yarn.scheduler.minimum-allocation-mb</name>
        <value>512</value>
    </property>
    
    <property>
        <name>yarn.scheduler.maximum-allocation-mb</name>
        <value>3072</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.vmem-check-enabled</name>
        <value>false</value>
    </property>
</configuration>
EOF
```

#### 6.5 mapred-site.xml
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
        <value>1024</value>
    </property>
    
    <property>
        <name>mapreduce.reduce.memory.mb</name>
        <value>1024</value>
    </property>
</configuration>
EOF
```

#### 6.6 workers file
```bash
cat > $HADOOP_HOME/etc/hadoop/workers << 'EOF'
datanode1
datanode2
datanode3
EOF
```

### Step 7: Distribute Cluster SSH Key to DataNodes
```bash
# Create cluster key distribution script
cat > /tmp/setup_datanode_ssh.sh << 'EOF'
#!/bin/bash
DATANODE_IP=$1
echo "=== Setting up SSH for DataNode: $DATANODE_IP ==="

# Copy cluster SSH key to DataNode
scp /root/.ssh/hadoop_cluster.pub root@$DATANODE_IP:/tmp/cluster_key.pub

# Setup SSH on DataNode
ssh root@$DATANODE_IP << 'REMOTE_EOF'
# Create SSH directory
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Add cluster key to authorized_keys
cat /tmp/cluster_key.pub >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
chown -R root:root /root/.ssh

# Install cluster key as default
cp /tmp/cluster_key.pub /root/.ssh/id_rsa.pub

# Generate matching private key (NameNode will provide)
echo "SSH setup complete on $(hostname)"
rm /tmp/cluster_key.pub
REMOTE_EOF

echo "✅ DataNode $DATANODE_IP SSH setup complete"
EOF

chmod +x /tmp/setup_datanode_ssh.sh

# Setup SSH for all DataNodes (run after DataNodes are accessible)
echo "=== SSH Key Distribution Ready ==="
echo "Run these commands after DataNodes are accessible:"
echo "/tmp/setup_datanode_ssh.sh 192.168.192.2"
echo "/tmp/setup_datanode_ssh.sh 192.168.192.3" 
echo "/tmp/setup_datanode_ssh.sh 192.168.192.4"
```

### Step 8: Firewall Configuration
```bash
# Open required ports
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 9000/tcp   # NameNode
sudo ufw allow 9870/tcp   # NameNode Web UI
sudo ufw allow 9868/tcp   # Secondary NameNode
sudo ufw allow 8088/tcp   # ResourceManager Web UI
sudo ufw allow 8030/tcp   # ResourceManager
sudo ufw allow 8031/tcp   # ResourceManager Scheduler
sudo ufw allow 8032/tcp   # ResourceManager Tracker
sudo ufw allow 8033/tcp   # ResourceManager Admin

# Enable firewall
sudo ufw --force enable
sudo ufw status
```

### Step 8: Format NameNode
```bash
# Format NameNode (ONLY run once!)
hdfs namenode -format -force

# Check formatting success
ls -la $HADOOP_HOME/data/namenode/current/
```

### Step 9: Start Services
```bash
# Stop any existing services first
$HADOOP_HOME/sbin/stop-all.sh
sleep 5

# Start HDFS services
$HADOOP_HOME/sbin/start-dfs.sh

# Wait for services to start
sleep 15

# Start YARN services
$HADOOP_HOME/sbin/start-yarn.sh

# Wait for services to start
sleep 10

# Note: DataNode errors are normal if DataNodes haven't been setup yet
# Only NameNode services should start on this node initially
```

### Step 10: Verification
```bash
# 1. Check running processes
jps

# Expected output on NameNode:
# - NameNode ✅
# - SecondaryNameNode ✅
# - ResourceManager ✅
# Note: DataNode/NodeManager will start after DataNodes are setup

# 2. Check HDFS status
hdfs dfsadmin -report

# 3. Check YARN nodes
yarn node -list

# 4. Web UI access - Check public IP first
PUBLIC_IP=$(curl -s -4 icanhazip.com)
echo "=== Web UI Access URLs ==="
echo "NameNode Web UI: http://$PUBLIC_IP:9870"
echo "ResourceManager Web UI: http://$PUBLIC_IP:8088"
echo ""
echo "Alternative (ZeroTier network only):"
echo "NameNode Web UI: http://192.168.192.1:9870"
echo "ResourceManager Web UI: http://192.168.192.1:8088"
```

## 🔧 Troubleshooting

### Common Issues:

#### 1. **NameNode fails to start:** 
```bash
# Check logs
tail -20 $HADOOP_HOME/logs/hadoop-root-namenode-*.log

# Check if formatted
ls -la $HADOOP_HOME/data/namenode/current/
```

#### 2. **Web UI not accessible externally:**
```bash
# Check service binding
ss -tlnp | grep -E "(9870|8088)"

# Should show: 0.0.0.0:9870 (all interfaces)
# Not: 192.168.192.1:9870 (ZeroTier only)

# Get public IP
PUBLIC_IP=$(curl -s -4 icanhazip.com)
echo "Try: http://$PUBLIC_IP:9870"

# Test from VPS
curl -I http://localhost:9870
curl -I http://$PUBLIC_IP:9870
```

#### 3. **Services bind to wrong interface:**
```bash
# Fix binding to all interfaces
$HADOOP_HOME/sbin/stop-all.sh

# Update configs
sed -i 's/192.168.192.1:9870/0.0.0.0:9870/g' $HADOOP_HOME/etc/hadoop/hdfs-site.xml
sed -i 's/192.168.192.1:8088/0.0.0.0:8088/g' $HADOOP_HOME/etc/hadoop/yarn-site.xml

# Restart services
$HADOOP_HOME/sbin/start-all.sh
```

#### 4. **Hadoop 3.x user variable errors:**
```bash
# Add required user variables
echo 'export HDFS_NAMENODE_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export HDFS_DATANODE_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export HDFS_SECONDARYNAMENODE_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export YARN_RESOURCEMANAGER_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
echo 'export YARN_NODEMANAGER_USER="root"' >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh
```

### Log Files:
```bash
# NameNode logs
tail -f $HADOOP_HOME/logs/hadoop-root-namenode-*.log

# ResourceManager logs  
tail -f $HADOOP_HOME/logs/hadoop-root-resourcemanager-*.log

# List all logs
ls -la $HADOOP_HOME/logs/
```

## ✅ Success Criteria

### NameNode Setup Complete:
- [x] NameNode process running
- [x] ResourceManager process running  
- [x] Secondary NameNode running
- [x] Web UIs accessible via public IP
- [x] Services bind to correct interfaces
- [x] Firewall configured properly

### Pending (After DataNodes Setup):
- [ ] All 3 DataNodes connected
- [ ] HDFS replication working
- [ ] YARN NodeManagers registered
- [ ] MapReduce jobs can be submitted

## 🌐 Web UI Access

### **Primary Access (Public IP):**
```bash
# Get your public IP
PUBLIC_IP=$(curl -s -4 icanhazip.com)
echo "NameNode Web UI: http://$PUBLIC_IP:9870"
echo "ResourceManager Web UI: http://$PUBLIC_IP:8088"
```

**Example URLs:**
- **NameNode UI:** http://146.190.95.21:9870
- **ResourceManager UI:** http://146.190.95.21:8088

### **Alternative Access (ZeroTier Network):**
- **NameNode UI:** http://192.168.192.1:9870  
- **ResourceManager UI:** http://192.168.192.1:8088
- *Note: Only accessible from devices in ZeroTier network*

### **Web UI Features:**

#### **NameNode UI (Port 9870):**
- HDFS overview and statistics
- DataNode status and health
- File system browser
- Block information and locations
- NameNode logs and metrics

#### **ResourceManager UI (Port 8088):**
- YARN applications and jobs
- NodeManager status
- Queue management
- Resource allocation
- Application history

---

**🎯 NameNode is the brain of your Hadoop cluster - Web UIs provide complete cluster visibility!**
