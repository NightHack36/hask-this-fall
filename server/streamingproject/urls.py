
from django.conf.urls import url
from django.contrib import admin

from streamingproject import views
from django.conf.urls.static import static
from django.conf import settings 
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

# url patterns for browsing 

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^stream/(?P<num>\d+)/(?P<stream_path>(.*?))/$', views.dynamic_stream, name="dynamic_stream"),
    url(r'^stream/screen/$', views.indexscreen),
    url(r'^mes/.*/$', views.changeline),
    url(r'^info/', views.getline),
    url(r'^end/', views.end),
    url(r'^play/.*', views.playvid),
    url(r'^start/', views.start),
    url(r'^profile', views.profilepage),
    url(r'', views.startpage)
    
    
]
urlpatterns += staticfiles_urlpatterns()