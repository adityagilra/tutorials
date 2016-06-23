# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:25:11 2014

@author: aditya
"""

class dog:
    def __init__(self,breed,age,color):
        self.pedigree = breed
        self.age = age
        self.color = color
    
    def bark(self):
        if self.age<5:
            print "woof"
        else:
            print "hoo"

class domesticated_dog(dog):
    def __init__(self,owner,*args):
        self.owner = owner
        dog.__init__(self,*args)
        
    def bark(self):
        dog.bark(self)
        dog.bark(self)

if __name__=="__main__":
    labrador1 = dog('labrador',12,'black')
    print labrador1.pedigree
    print labrador1.age
    labrador2 = dog('labrador',3,'black-white')
    print labrador2.pedigree
    print labrador2.age
    print labrador2.color
    labrador1.bark()
    labrador2.bark()
    
    pomeranian = domesticated_dog('Adi','pomeranian',2,'white')
    print pomeranian.owner
    pomeranian.bark()
    

    