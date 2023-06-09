from abc import ABC, abstractmethod
import ftplib
import pickle
import os

class DataSaver(ABC):
    print : bool = False

    @abstractmethod 
    def save(self, localName:str, remoteName:str) -> None: pass

    @abstractmethod
    def load(self, localName:str, remoteName:str) -> None: pass


class FTPDataSaver(DataSaver):
    def __init__(self, host:str, user:str, password:str, chdir: str='.' ):
        super().__init__()
        self.credentials = host, user, password
        self.chdir = chdir


    def _ftp(self):
        # Return new FTP connection object
        ftp = ftplib.FTP(*self.credentials)
        ftp.encoding = "utf-8"
        return ftp


    def mkdir(self, remoteName:str) -> None:
        try:
            self._ftp().mkd(remoteName)
        except:
            if self.print:
                print(f'Cannot mkdir `ftp:{remoteName}`')


    def save(self, localName:str, remoteName:str) -> None:
        try:
            with open(localName, "rb") as file:
                self._ftp().storbinary(f"STOR {self.chdir + '/' + remoteName}", file)
        except:
            if self.print:
                print(f'Error while saving `{localName}` to `ftp:{self.chdir}/{remoteName}`')
        else:
            if self.print:
                print(f'Uploaded `ftp:{self.chdir}/{remoteName}`')


    def load(self, localName:str, remoteName:str) -> None:
        try:
            with open(localName, "wb") as file:
                self._ftp().retrbinary(f'RETR {self.chdir}/{remoteName}', file.write)
        except:
            if self.print:
                print(f'Cannot download `ftp:{f}` to `{localName}`')                
        else:
            if self.print:
                print(f'Downloaded `ftp:{f}` to `{localName}`')
