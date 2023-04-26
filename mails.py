import smtplib #importing the module

def mailer(links,website):
    sender_add='reuben.sj2003@gmail.com' #storing the sender's mail id
    receiver_add='akilesh0876@gmail.com' #storing the receiver's mail id
    password='ncjvvdqsmydfukrw' #storing the password to log in

    #creating the SMTP server object by giving SMPT server address and port number
    smtp_server=smtplib.SMTP("smtp.gmail.com",587)
    smtp_server.ehlo() #setting the ESMTP protocol

    smtp_server.starttls() #setting up to TLS connection
    smtp_server.ehlo() #calling the ehlo() again as encryption happens on calling startttls()

    smtp_server.login(sender_add,password) #logging into out email id
    msg_to_be_sent =f"{links} are the links detected to have obscenity. The website where we found these links are from {website}"


    #sending the mail by specifying the from and to address and the message 
    smtp_server.sendmail(sender_add,receiver_add,msg_to_be_sent)
    smtp_server.quit()#terminating the server

    return 'The mail is sent.'