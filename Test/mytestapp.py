import interface
import wx 

class GetBut(interface.MyFrame): 
   def __init__(self,parent): 
      interface.MyFrame.__init__(self,parent)  
      
   def initialize(self):
        self.Show(True)	
        
   def ShowIcon(self,e): 
      print e.GetId()
      #cv2.putText(frame,'%d' %e.id(),(x,y), font, 2,(255,255,255),2,cv2.LINE_AA)
      
app = wx.App(False)
win=GetBut(None)
win.Show(True)
app.MainLoop() 