# Mini LLMOps Service

Bu layihədə production-ready LLM çağırışı üçün timeout, retry və fallback mexanizmləri tətbiq edilmişdir.

Hər uğurlu LLM çağırışının token istifadəsi və xərci costs.csv faylında qeyd olunur. Eyni sual ikinci dəfə verildikdə cache istifadə olunur və LLM API çağırışı edilmir.

Layihədə cache və xərc hesablama funksiyaları üçün pytest testləri yazılmışdır. GitHub Actions hər push zamanı testləri avtomatik işə salır.