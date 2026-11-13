"""Diagnostic producer-side output wait; original consumer and buffers retained."""
import functools

def install(output_class,capture):
    old=output_class.__init__
    @functools.wraps(old)
    def init(self,*args,**kwargs):
        old(self,*args,**kwargs)
        # The original init has already recorded this event after pinned D2H.
        # Only wait placement changes; get_output still performs its native wait.
        c=capture()
        with c.span('diagnostic_output_producer_wait'):
            self.copy_event.synchronize()
    output_class.__init__=init
    return old
