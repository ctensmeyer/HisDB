import sys
import os

INDEX_TEMPLATE = """
<html>
    <table>
        {}
    </table>
    <body>
        {}
    </body>
</html>
"""

OTHER_ITEM_TEMPLATE = """
<tr>
<td>
{}
</td>
<td>
{}
</td>
</tr>
"""

INDEX_ITEM_TEMPLATE = """
<a href="{}">{}, {}, {}, {}</a><br>
"""

INDIVIDUAL_PAGE_TEMPLATE = """
    <table>
        {}
    </table>
    <img src="{}" width=1000/>
"""

def make_individual_webpage(filename, display_img_path, headers, scores):
    other_items = []
    for h, s in zip(headers, scores):
        other_items.append(OTHER_ITEM_TEMPLATE.format(h,s))

    text = INDIVIDUAL_PAGE_TEMPLATE.format("".join(other_items), display_img_path)

    with open(filename, 'w') as f:
        f.write(text)

def gen_webpage(website_dir, display_img_paths, scores):
    if not os.path.exists(website_dir):
        os.makedirs(website_dir)

    headers = scores['individual_results'][0]
    results = scores['individual_results'][1:]


    for r, w in zip(results, display_img_paths):
        if r[2] == "X":
            r[2] = -1
        r.append(w)

    results.sort(key=lambda x: float(x[2]))

    all_items = []
    all_items.append(INDEX_ITEM_TEMPLATE.format("#", *headers[:4]))
    for i, result in enumerate(results):

        filename_local = str(i)+".html"
        filename = os.path.join(website_dir, filename_local)

        make_individual_webpage(filename, result[-1], headers[:4], result[:4])

        item = INDEX_ITEM_TEMPLATE.format(filename_local, *result[:4])
        all_items.append(item)

    other_items = []
    for k, v in scores.iteritems():
        if k=="individual_results":
            continue
        other_items.append(OTHER_ITEM_TEMPLATE.format(k,v))

    index_text = INDEX_TEMPLATE.format("".join(other_items), "".join(all_items))

    with open(os.path.join(website_dir, 'index.html'), 'w') as f:
        f.write(index_text)
