# -*- coding: utf-8 -*-
import wx
import wx.xrc

###########################################################################
## Class MyFrame
###########################################################################

class MyFrame ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Face Detection", pos = wx.Point( 250,250 ), size = wx.Size( 205,276 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Moods" ), wx.VERTICAL )
		
		self.but1 = wx.Button( sbSizer2.GetStaticBox(), 1, u"Sad", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer2.Add( self.but1, 0, wx.ALL|wx.EXPAND, 5 )
		
		self.but2 = wx.Button( sbSizer2.GetStaticBox(), 2, u"Happy", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer2.Add( self.but2, 0, wx.ALL|wx.EXPAND, 5 )
		
		self.but3 = wx.Button( sbSizer2.GetStaticBox(), 3, u"Angry", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer2.Add( self.but3, 0, wx.ALL|wx.EXPAND, 5 )
		
		self.but4 = wx.Button( sbSizer2.GetStaticBox(), 4, u"Sleepy", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer2.Add( self.but4, 0, wx.ALL|wx.EXPAND, 5 )
		
		self.but5 = wx.Button( sbSizer2.GetStaticBox(), 5, u"Indifferent", wx.Point( -10,-1 ), wx.DefaultSize, 0 )
		sbSizer2.Add( self.but5, 0, wx.ALL|wx.EXPAND, 5 )
		
		#self.but6 = wx.Button( sbSizer2.GetStaticBox(), 6, u"Quit", wx.DefaultPosition, wx.DefaultSize, 0 )
		#sbSizer2.Add( self.but6, 0, wx.ALL|wx.EXPAND, 5 )
		
				
		self.SetSizer( sbSizer2 )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.but1.Bind( wx.EVT_BUTTON, self.ShowIcon,id=self.but1.GetId() )
		self.but2.Bind( wx.EVT_BUTTON, self.ShowIcon,id=self.but2.GetId() )
		self.but3.Bind( wx.EVT_BUTTON, self.ShowIcon,id=self.but3.GetId() )
		self.but4.Bind( wx.EVT_BUTTON, self.ShowIcon,id=self.but4.GetId() )
		self.but5.Bind( wx.EVT_BUTTON, self.ShowIcon,id=self.but5.GetId() )
		#self.but6.Bind( wx.EVT_BUTTON, self.Quit)
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def ShowIcon( self, event ):
		event.Skip()
	def Quit( self, event ):
		event.Skip()
	
	
	
	

