import app.main as shop


def test_search_respects_budget_and_category():
    res = shop.search_products('espresso grinder', category='grinders', max_price=200)
    assert res and all(p['price'] <= 200 and p['category'] == 'grinders' for p in res)


def test_order_needs_email():
    r = shop.check_order('AL-20481')
    assert r['found'] is False and 'email' in r['message'].lower()


def test_order_wrong_email_reveals_nothing():
    r = shop.check_order('AL-20481', 'someone.else@example.com')
    assert r == {'found': False, 'message': 'No order matches that order number and email combination.'}


def test_order_found_without_leaking_email():
    r = shop.check_order('al-20481', 'MARIA.LOPEZ@example.com')
    assert r['found'] and r['status'] == 'shipped' and 'email' not in r


def test_policy_lookup():
    assert 'restocking fee' in shop.get_policy('returns')['policy']
    assert 'topics_available' in shop.get_policy('unknown topic')


def test_lead_validation_and_dedupe(tmp_path, monkeypatch):
    monkeypatch.setattr(shop, 'LEADS', str(tmp_path / 'leads.json'))
    assert shop.save_lead('Ann', 'not-an-email', 'quote')['saved'] is False
    first = shop.save_lead('Ann', 'ann@example.com', 'Quote for Lume Duo')
    second = shop.save_lead('Ann', 'ann@example.com', 'quote for lume duo')
    assert first['saved'] and second['lead_id'] == first['lead_id']
    assert len(shop._load(shop.LEADS, [])) == 1
