# -*- coding: utf-8 -*-
"""把当周 Top-50 报告发到邮箱。

凭证从环境变量读取, 绝不写进代码或仓库:
    SMTP_HOST   默认 smtp.qq.com
    SMTP_PORT   默认 465 (SSL)
    SMTP_USER   发信邮箱
    SMTP_PASS   **授权码**, 不是登录密码 (QQ邮箱: 设置->账号->开启SMTP服务后获取)
    MAIL_TO     收件人, 多个用逗号分隔

放在项目根的 .env 里(已在 .gitignore 中)。缺任一项则跳过发信并明确报错,
不静默失败——静默失败的定时任务等于没有。

用法:
    python src/live/send_report.py                  # 发最新一期
    python src/live/send_report.py --asof 2026-08-31
    python src/live/send_report.py --test           # 只发一封连通性测试信
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import PROJ_ROOT as PROJ, OUTPUT_DIR

import os, ssl, glob, smtplib, argparse
from html import escape as _esc
from email.message import EmailMessage


def _cfg():
    miss = [k for k in ('SMTP_USER', 'SMTP_PASS', 'MAIL_TO') if not os.environ.get(k)]
    if miss:
        raise SystemExit(f'缺少环境变量 {miss}——请在项目根 .env 中配置后重试。\n'
                         f'注意 SMTP_PASS 填的是邮箱**授权码**, 不是登录密码。')
    return {'host': os.environ.get('SMTP_HOST', 'smtp.qq.com'),
            'port': int(os.environ.get('SMTP_PORT', '465')),
            'user': os.environ['SMTP_USER'], 'pw': os.environ['SMTP_PASS'],
            'to': [x.strip() for x in os.environ['MAIL_TO'].split(',') if x.strip()]}


def send(subject, body, attachments=(), c=None, html=None):
    c = c or _cfg()
    m = EmailMessage()
    m['Subject'] = subject
    m['From'] = c['user']
    m['To'] = ', '.join(c['to'])
    m.set_content(body)
    if html:
        m.add_alternative(html, subtype='html')
    for p in attachments:
        if not os.path.exists(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        main, sub = ('image', 'png') if ext == '.png' else ('text', 'csv')
        with open(p, 'rb') as f:
            data = f.read()
        cid = os.path.basename(p)
        if main == 'image':
            m.add_attachment(data, maintype=main, subtype=sub, filename=cid,
                             cid=f'<{cid}>')
        else:
            m.add_attachment(data, maintype=main, subtype=sub, filename=cid)
    ctx = ssl.create_default_context()
    if c['port'] == 465:
        with smtplib.SMTP_SSL(c['host'], c['port'], context=ctx, timeout=30) as s:
            s.login(c['user'], c['pw'])
            s.send_message(m)
    else:
        with smtplib.SMTP(c['host'], c['port'], timeout=30) as s:
            s.starttls(context=ctx)
            s.login(c['user'], c['pw'])
            s.send_message(m)
    print(f'已发送至 {m["To"]}: {subject}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asof', default=None)
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    c = _cfg()
    if a.test:
        send('[联动锚定策略] SMTP 连通性测试',
             '这是一封测试信。收到即说明定时任务可以把每周 Top-50 发到这里。', c=c)
        return
    live = f'{OUTPUT_DIR}/live'
    if a.asof:
        txt = f'{live}/top{a.top}_{a.asof}.txt'
    else:
        cand = sorted(glob.glob(f'{live}/top{a.top}_*.txt'))
        if not cand:
            raise SystemExit(f'{live} 下没有 top{a.top}_*.txt, 请先跑 weekly_top50.py')
        txt = cand[-1]
    ds = os.path.basename(txt).replace(f'top{a.top}_', '').replace('.txt', '')
    body = open(txt, encoding='utf-8').read()

    # 图表: 缺图不算失败, 文本与 CSV 仍要送到
    pngs = []
    try:
        import make_charts
        _, pngs = make_charts.build(ds, a.top)
    except Exception as e:
        print(f'警告: 图表生成失败({type(e).__name__}: {e}), 仅发文本与 CSV')

    imgs = ''.join(
        f'<h3 style="font:600 15px/1.5 system-ui;color:#222;margin:26px 0 8px">{t}</h3>'
        f'<img src="cid:{os.path.basename(p)}" style="max-width:100%;border:1px solid #e3e3e3;border-radius:4px">'
        for p, t in zip(pngs, ['打分排名 Top-50(含名称与三块得分构成)',
                               '目标持仓权重分布 + 权重最大的 40 只',
                               '行业分布与偏离度',
                               '纸上交易累计净值']))
    html = (f'<div style="font:14px/1.6 system-ui,-apple-system,sans-serif;'
            f'color:#222;max-width:1100px">'
            f'<pre style="font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;'
            f'background:#fafafa;border:1px solid #eee;border-radius:4px;'
            f'padding:14px;overflow-x:auto">{_esc(body)}</pre>{imgs}'
            f'<p style="color:#888;font-size:12px;margin-top:26px">'
            f'CSV 明细见附件。执行按路径B: T+1 11:30 前卖出腿, 13:30-14:30 买入腿。</p>'
            f'</div>')

    atts = [f'{live}/top{a.top}_{ds}.csv', f'{live}/portfolio_{ds}.csv',
            f'{live}/orders_{ds}.csv'] + pngs
    send(f'[联动锚定策略] {ds} 打分 Top-{a.top}', body, atts, c=c, html=html)


if __name__ == '__main__':
    main()
