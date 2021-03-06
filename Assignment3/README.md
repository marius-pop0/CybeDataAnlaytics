# CTU-Malware-Capture-Botnet-46 or Scenario 5 in the CTU-13 dataset.
# Description
- Probable Name: Virut
- MD5: 85f9a5247afbe51e64794193f1dd72eb
- SHA1: 532f81d162ddb3b3563bbb942f7eb3220f0f604d
- SHA256: e8c63b3753504ff27cb93555c62beaaded5f3e485017d950f0aa0831a9c45ac8
- Password of zip file: infected
- Duration: 

- [VirusTotal](https://www.virustotal.com/en/file/e8c63b3753504ff27cb93555c62beaaded5f3e485017d950f0aa0831a9c45ac8/analysis/)
- [HybridAnalysis](https://www.hybrid-analysis.com/sample/e8c63b3753504ff27cb93555c62beaaded5f3e485017d950f0aa0831a9c45ac8?environmentId=2)
- RobotHash

[![](https://robohash.org/85f9a5247afbe51e64794193f1dd72eb)](https://robohash.org)

# Files

- capture20110815-2.pcap

    It is a pcap capture with __all__ the traffic (background, normal and botnet)

    This pcap file was not made public because it contains too much private information about the users of the network.

    This file was captures on the main router of the University network. 

- botnet-capture-20110815-2-fast-flux.pcap

    Capture with only the botnet traffic. It is made public.

    This file was captured on the interface of the virtual machine being infected. 

- capture20110815-2.pcap.netflow.labeled

    This file has the netflows generated by a __unidirectional__ argus. The labels were assigned as this:

        - First put Background to all the flows.
        - Put LEGITIMATE to the flows that match some filters.
        - Put Botnet to the flows that come to or from the infected IP addresses

- bro
    - Folder with all the bro output files

- detailed-bidirectional-flow-labels
    - Folder with the bidirectional flows. These are the files you should use for your research. They have better labels and better quality of data.

- *.html
    - This is an html graphical page made with CapTipper of the HTTP requests in the capture.

- *.json
    - File needed by Captipper for the html page.

- *truncated.pcap.bz2
    - This is a truncated pcap file of the __complete__ capture. The pcap is truncated to have only the following starting bytes for each packet:
        - TCP: 54 bytes
        - UDP: 42 bytes
        - ICMP: 66 bytes
    - See [this description](https://stratosphereips.org/new-dataset-ctu-13-extended-now-includes-pcap-files-of-normal-traffic.html) of the truncation.

# IP Addresses
    - Infected hosts
        - 147.32.84.165: Windows XP English version Name: SARUMAN. Label: Botnet. Amount of bidirectional flows: 1802
    - Normal hosts:
        - 147.32.84.170 (amount of bidirectional flows: 3620, Label: Normal-V42-Stribrek)
        - 147.32.84.134 (amount of bidirectional flows: 2214, Label: Normal-V42-Jist)
        - 147.32.84.164 (amount of bidirectional flows: 3444, Label: Normal-V42-Grill)
        - 147.32.87.36 (amount of bidirectional flows: 28, Label: CVUT-WebServer. This normal host is not so reliable since is a webserver)
        - 147.32.80.9 (amount of bidirectional flows: 10, Label: CVUT-DNS-Server. This normal host is not so reliable since is a dns server)
        - 147.32.87.11 (amount of bidirectional flows: 11, Label: MatLab-Server. This normal host is not so reliable since is a matlab server)

## Important Label note
Please note that the labels of the flows generated by the malware start with "From-Botnet". The labels "To-Botnet" are flows sent to the botnet by unknown computers, so they should not be considered malicious perse.
Also for the normal computers, the counts are for the labels "From-Normal". The labels "To-Normal" are flows sent to the botnet by unknown computers, so they should not be considered malicious perse.

# Timeline
## Mon Aug 15 16:43:26 CEST 2011
We started the overall capture of the department.

We are going to infect the VM with a fast-flux malware.
Bandwith will be at 100kBps.

## Mon Aug 15 16:52:23 CEST 2011
We started the capture.

## Mon Aug 15 17:13:07 CEST 2011
We stopped the VM and the bot capture and the overall capture.

# Traffic Analysis


# Disclaimer 

    These files were generated in the Stratosphere Lab as part of the Malware Capture Facility Project in the CVUT University, Prague, Czech Republic.
    The goal is to store long-lived real botnet traffic and to generate labeled netflows files.
    Any question feel free to contact us:
    Sebastian Garcia: sebastian.garcia@agents.fel.cvut.cz
    You are free to use these files as long as you reference this project and the authors as follows:
    Garcia, Sebastian. Malware Capture Facility Project. Retrieved from https://stratosphereips.org
