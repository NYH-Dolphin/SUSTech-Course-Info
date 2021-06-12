import random
import socket
import time
import dnslib

# Basic message
# ————————————————————————————————————————————————————————————————
# Server
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
port = 12000
server.bind(('', port))
# Forward query server
farServer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Cache and Database
cache = {}
database = [('a.root-servers.net', '198.41.0.4'),
            ('b.root-servers.net', '199.9.14.201'),
            ('c.root-servers.net', '192.33.4.12'),
            ('d.root-servers.net', '199.7.91.13'),
            ('e.root-servers.net', '192.203.230.10'),
            ('f.root-servers.net', '192.5.5.241'),
            ('g.root-servers.net', '192.112.36.4'),
            ('h.root-servers.net', '198.97.190.53'),
            ('i.root-servers.net', '192.36.148.17'),
            ('j.root-servers.net', '192.58.128.30'),
            ('k.root-servers.net', '193.0.14.129'),
            ('l.root-servers.net', '199.7.83.42'),
            ('m.root-servers.net', '202.12.27.33')]


# ————————————————————————————————————————————————————————————————

# get DNS responce format from cache_in byte format
def from_cache_get_DNS_format(qname, qtype, qheader):
    res = dnslib.DNSRecord(
        dnslib.DNSHeader(id=qheader.id, q=qheader.q, r=cache[qname][qtype][2], auth=cache[qname][qtype][3],
                         ra=cache[qname][qtype][4]), q=dnslib.DNSQuestion(qname, qtype), rr=cache[qname][qtype][5],
        auth=cache[qname][qtype][6], ar=cache[qname][qtype][7])
    res.header.qr = 1
    cacheMessage = bytes(res.pack())
    return cacheMessage


# Boolean_whether in cached
def in_cached(qname, qtype):
    # check cached
    if qname not in cache:
        cache[qname] = {}
    if qtype not in cache[qname]:
        return False

    return checkTime(qname, qtype)


# Boolean_whether in database
def in_database(qname):
    for db in database:
        if qname == db[0]:
            return True

    return False


# Boolean_whether Time is enough
def checkTime(qname, qtype):
    content = cache[qname][qtype]
    now_time = (time.time())
    if now_time - content[0] >= content[1]:
        return False
    else:
        return True


# add message to cache
def add_to_cache(qname, qtype, msg):
    record = dnslib.DNSRecord.parse(msg)

    # calculate the minimum ttl
    cur_t = int(time.time())  # record current time
    ttl = 1000000000000000
    for rr in record.rr:
        ttl = min(ttl, rr.ttl)

    # store all information in cache
    cache_format(cur_t, ttl, record, qname, qtype)


def cache_format(cur_t, ttl, record, qname, qtype):
    cache[str(qname)][qtype] = [cur_t, ttl, record.header.a, record.header.auth, record.header.ar,
                                record.rr, record.auth, record.ar]


class IterativeQuery(object):
    @staticmethod
    def start(msg, req):
        print('From iterative query')
        # forward query to a public dns server
        GoogleDNS, GooglePort = '8.8.8.8', 53
        farServer.sendto(msg, (GoogleDNS, GooglePort))
        msg = farServer.recv(2048)
        # add new response to cache
        add_to_cache(str(req.q.qname), req.q.qtype, msg)
        # use cache to create answer
        resmsg = from_cache_get_DNS_format(str(req.q.qname), req.q.qtype, req.header)
        return resmsg


class RecursiveQuery(object):
    @staticmethod
    def start(msg, req):
        global resmsg
        print('From recursive query')
        # Randomly pick one root DNS server
        randomAddrNum = random.randint(0, 12)
        recursiveSerer = database[randomAddrNum][1]
        # Create temp request message and set rd = 0
        tempReq = dnslib.DNSRecord.parse(msg)
        tempReq.header.rd = 0
        msg = bytes(tempReq.pack())
        # Send message to one root DNS server
        tmp = []
        while True:

            # Forward query to next server
            farServer.sendto(msg, (recursiveSerer, 53))
            msg2 = farServer.recv(2048)
            tempReq = dnslib.DNSRecord.parse(msg2)

            # The query is refused, start from another root DNS server
            if tempReq.header.a == 0 and tempReq.header.auth == 0:
                # Pick another DNS server
                while True:
                    randomAddrNum2 = random.randint(0, 12)
                    if randomAddrNum2 != randomAddrNum:
                        randomAddrNum = randomAddrNum2
                        break
                    else:
                        continue
                recursiveSerer = database[randomAddrNum][1]
                continue

            # The query is accept
            if tempReq.header.a != 0:
                # The name is not canonical
                if tempReq.header.a == 1 and req.q.qtype == 1 and tempReq.rr[0].rtype == 5:
                    tempReq2 = dnslib.DNSRecord.parse(msg)
                    tempReq2.q.qname = dnslib.DNSLabel(str(tempReq.rr[0].rdata))
                    msg = bytes(tempReq2.pack())
                    # record CNAME
                    tmp.append(tempReq.rr[0])
                    continue
                # CNAME RR + 1

                tempReq2 = dnslib.DNSRecord.parse(msg2)
                tempReq2.header.a += 1
                tmp.extend(tempReq2.rr)
                tempReq2.rr = tmp
                # add to cache
                msg2 = bytes(tempReq2.pack())
                add_to_cache(str(req.q.qname), req.q.qtype, msg2)
                resmsg = from_cache_get_DNS_format(str(req.q.qname), req.q.qtype, req.header)
                break
            recursiveSerer = str(tempReq.auth[random.randint(0,tempReq.header.auth - 1)].rdata)
        return resmsg


class DNSServer(object):
    @staticmethod
    def start():
        while True:
            # receive query from client
            msg, addr = server.recvfrom(2048)
            # Make query's format as DNSRecord
            req = dnslib.DNSRecord.parse(msg)
            # Print query
            print('Query [name=%s, type=%s]' % (
                str(req.q.qname), dnslib.QTYPE.get(req.q.qtype)))
            # set response message
            resmsg = b'null'
            # in cached
            if in_cached(str(req.q.qname), req.q.qtype):
                print('From cache')
                resmsg = from_cache_get_DNS_format(str(req.q.qname), req.q.qtype, req.header)  # get reply from cache
                server.sendto(resmsg, addr)
            # in database(search for root server's ip)
            if in_database(req.q.qname) and not in_cached(str(req.q.qname), req.q.qtype):
                print('From database')
                add_to_cache(str(req.q.qname), req.q.qtype, msg)
                resmsg = from_cache_get_DNS_format(str(req.q.qname), req.q.qtype, req.header)
                server.sendto(resmsg, addr)
            # Neither in cache nor in database
            if not in_cached(str(req.q.qname), req.q.qtype) and not in_database(req.q.qname):
                if req.header.rd == 0:
                    resmsg = IterativeQuery.start(msg, req)
                if req.header.rd == 1:
                    resmsg = RecursiveQuery.start(msg, req)
            # Response to client
            server.sendto(resmsg, addr)


if __name__ == '__main__':
    print('local DNSSever start at: 127.0.0.1:' + str(port))
    DNSServer.start()
