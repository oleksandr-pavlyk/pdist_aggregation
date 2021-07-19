#pragma once

// Heap internal structure is 0-based array of n-elements
// Values are compared using comparator comp.
// left-child of element at i is at 2 * i + 1
// right-child of element at i is at 2 * i + 2

template <typename accessorT, typename valT, typename Comp>
void heap_sift_up(accessorT a, size_t n, Comp &comp)
{
    if (n > 1)
    {
        size_t last = (n - 1);
        n = (n / 2) - 1;
        size_t i = n;
        if (comp(a[i], a[last]))
        {
            valT v = a[last];
            do
            {
                a[last] = a[i];
                last = i;
                if (n == 0)
                {
                    break;
                }
                n = (n - 1) / 2;
                i = n;
            } while (comp(a[i], v));
            a[last] = v;
        }
    }
}

template <typename accessorT, typename valT, typename Comp>
void heap_sift_down(accessorT a, size_t n, size_t start, Comp &comp)
{
    // go down the tree from node at start
    // fixing violations of heap order until
    // a node encountered at which heap-order
    // is satisfied.
    auto child = start;

    if (n < 2 || (n - 2) / 2 < child)
    {
        return;
    }

    child = 2 * child + 1;
    size_t child_i = child;

    if ((child + 1) < n && comp(a[child_i], a[child_i + 1]))
    {
        // right-child exists and is greater than left-child
        ++child_i;
        ++child;
    }
    if (comp(a[child_i], a[start]))
    {
        // early exit, already in heap-order
        return;
    }

    valT top = a[start];
    do
    {
        // not in heap-order, swap the parent with it's largest child
        a[start] = a[child_i];

        start = child_i;
        if ((n - 2) / 2 < child)
        {
            break;
        }

        child = 2 * child + 1;
        child_i = child;
        if ((child + 1) < n && comp(a[child_i], a[child_i + 1]))
        {
            ++child_i;
            ++child;
        }

    } while (!(comp(a[child_i], top)));
    a[start] = top;
}

template <typename accessorT, typename valT, typename Comp>
void make_heap(accessorT a, size_t n, Comp &comp)
{
    if (n > 1)
    {
        for (auto start = n / 2; start > 0; --start)
        {
            heap_sift_down<accessorT, valT, Comp>(
                a, n, (start - 1), comp);
        }
    }
}

template <typename accessorT, typename valT, typename Comp>
void push_heap(accessorT a, size_t n, Comp &comp)
{
    heap_sift_up<accessorT, valT, Comp>(a, n, comp);
}

template <typename accessorT, typename valT, typename Comp>
void pop_heap(accessorT a, size_t n, Comp &comp)
{
    if (n > 1)
    {
        auto tail_index = n - 1;
        const auto tmp = a[0];
        a[0] = a[tail_index];
        a[tail_index] = tmp;

        heap_sift_down<accessorT, valT, Comp>(
            a, tail_index, 0, comp);
    }
}

template <typename accessorT, typename valT, typename Comp>
void sort_heap(accessorT a, size_t n, Comp &comp)
{
    if (n > 1)
    {
        for (auto i = n; i > 1; --i)
        {
            pop_heap<accessorT, valT, Comp>(a, i, comp);
        }
    }
}
