# ZeroTier Network Setup for Hadoop Cluster

## Network Configuration
- **NameNode**: 192.168.192.1
- **DataNode1**: 192.168.192.2  
- **DataNode2**: 192.168.192.3
- **DataNode3**: 192.168.192.4

## Step 1: Install ZeroTier on All 4 VPS

```bash
# Install ZeroTier
curl -s https://install.zerotier.com | sudo bash

# Join network (replace YOUR_NETWORK_ID)
sudo zerotier-cli join YOUR_NETWORK_ID

# Check status
sudo zerotier-cli status
```

## Step 2: Configure ZeroTier Dashboard

1. Go to https://my.zerotier.com
2. Authorize each device
3. Assign IPs:
   - NameNode → 192.168.192.1
   - DataNode1 → 192.168.192.2
   - DataNode2 → 192.168.192.3  
   - DataNode3 → 192.168.192.4

## Step 3: Test Connectivity

```bash
# From NameNode ping DataNodes
ping 192.168.192.2
ping 192.168.192.3
ping 192.168.192.4

# From each DataNode ping NameNode
ping 192.168.192.1
```

## Troubleshooting

```bash
# If connection fails
sudo systemctl restart zerotier-one
sudo zerotier-cli leave YOUR_NETWORK_ID
sudo zerotier-cli join YOUR_NETWORK_ID

# Check status
sudo zerotier-cli listnetworks
```

---
**Note**: Replace `YOUR_NETWORK_ID` with your actual ZeroTier network ID.
