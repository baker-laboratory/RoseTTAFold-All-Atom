set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
"Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
"Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
"Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
"Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}


"My Plugins"

"nerd tree to show file system in vim"
Plugin 'scrooloose/nerdtree'

"vimarline for the status bar at the bottom of the page
Plugin 'vim-airline/vim-airline'
Plugin 'vim-airline/vim-airline-themes'

"fonts for nerdtree
Plugin 'ryanoasis/vim-devicons'
set encoding=UTF-8

"Shows ctags on the side
Plugin 'majutsushi/tagbar'
map <C-l> :TagbarToggle<CR>

let g:tagbar_type_ansible = {
    \ 'ctagstype' : 'ansible',
    \ 'kinds' : [
        \ 't:tasks'
    \ ],
    \ 'sort' : 0
\ }

"Fuzzy finder
Plugin 'junegunn/fzf.vim'
Plugin 'junegunn/fzf'

"yaml
Plugin 'mrk21/yaml-vim'

"yaml folding
Plugin 'pedrohdz/vim-yaml-folds'

"Linter
"Plugin 'w0rp/ale'

"Airline
let g:airline#extensions#ale#enabled = 1

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line


" Maintainer: Ray YR Wang <ray.wangyr@gmail.com>
set nocompatible              " be iMproved

set backspace=indent,eol,start " allow backspacing over everything in insert mode

" When started as "evim", evim.vim will already have done these settings.
if v:progname =~? "evim"
  finish
endif

if has("vms")
  set nobackup		" do not keep a backup file, use versions instead
else
  set backup		" keep a backup file
endif
set history=1000		" keep 50 lines of command line history
set ruler		" show the cursor position all the time
set showcmd		" display incomplete commands
set incsearch		" do incremental searching

" For Win32 GUI: remove 't' flag from 'guioptions': no tearoff menu entries
" let &guioptions = substitute(&guioptions, "t", "", "g")

" Don't use Ex mode, use Q for formatting
map Q gq


" This is an alternative that also works in block mode, but the deleted
" text is lost and it only works for putting the current register.
"vnoremap p "_dp

" Switch syntax highlighting on, when the terminal has colors
" Also switch on highlighting the last used search pattern.
if &t_Co > 2 || has("gui_running")
  syntax on
  set hlsearch
endif

" Only do this part when compiled with support for autocommands.
if has("autocmd")

  " Enable file type detection.
  " Use the default filetype settings, so that mail gets 'tw' set to 72,
  " 'cindent' is on in C files, etc.
  " Also load indent files, to automatically do language-dependent indenting.
  filetype plugin indent on

  " Put these in an autocmd group, so that we can delete them easily.
  augroup vimrcEx
  au!

  " For all text files set 'textwidth' to 78 characters.
  " autocmd FileType text setlocal textwidth=78

  " When editing a file, always jump to the last known cursor position.
  " Don't do it when the position is invalid or when inside an event handler
  " (happens when dropping a file on gvim).
  autocmd BufReadPost *
    \ if line("'\"") > 0 && line("'\"") <= line("$") |
    \   exe "normal g`\"" |
    \ endif

  augroup END

else

  set autoindent		" always set autoindenting on

endif " has("autocmd")

" Convenient command to see the difference between the current buffer and the
" file it was loaded from, thus the changes you made.
command DiffOrig vert new | set bt=nofile | r # | 0d_ | diffthis
	 	\ | wincmd p | diffthis


"""""""""""""""""""""""""""""""""""""""""""""""
" I have no idea about what it is doing above "
"""""""""""""""""""""""""""""""""""""""""""""""

set hlsearch
set backspace=2 " make backspace work like most other apps

set autoindent
set smartindent

set number
set ruler
set showmode

set background=dark

set tabstop=4
set shiftwidth=4
set expandtab " replace tabs into spaces

nmap <silent> <C-N> :silent noh<CR>


let mapleader = ","

"let Tlist_Auto_Open = 1
"let Tlist_Show_One_File = 1
"map <silent> <leader>] :TlistToggle<cr>
"set foldenable
"autocmd FileType c,cpp,C,cc,CC,Cpp setl fdm=syntax | setl fen
"
"" mark setting
""""""""""""""""""""""""""""""
nmap <silent> <leader>hl <Plug>MarkSet
vmap <silent> <leader>hl <Plug>MarkSet
nmap <silent> <leader>hh <Plug>MarkClear
vmap <silent> <leader>hh <Plug>MarkClear
nmap <silent> <leader>hr <Plug>MarkRegex
vmap <silent> <leader>hr <Plug>MarkRegex

set tags+=tags

"call g++ compile file and replace the file name as .out"
nmap <C-c><C-c> :!g++ -Wall % -o %:r.out<CR>
nmap <tab> v>
nmap <s-tab> v<

vmap <tab> >gv
vmap <s-tab> <gv
"" CTRL-Tab is Next window, maximum size
"noremap <C-Tab> <C-W>w<C-W>_<C-W>\|
"inoremap <C-Tab> <C-O><C-W>w<C-O><C-W>_<C-W>\|
"cnoremap <C-Tab> <C-C><C-W>w<C-C><C-W>_<C-W>\|
"
"" F11 and F12 make split windows smaller/bigger
map <F12> <c-w>10+ <c-w>20>
map <F11> <c-w>10- <c-w>20<
"map <C-Down> <c-w> <Down>
"map <C-Up> <c-w> <Up>
"

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"  wangyr customized setup
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"========== programming useful tricks ==========
"folding settings"
set foldmethod=indent   "fold based on indent
set foldnestmax=10      "deepest fold is 10 levels
set nofoldenable        "dont fold by default
set foldlevel=1         "this is just what i use

"enable the plugin after setting up taglist"
filetype plugin on

"search upper directories util it finds tags
set tags=./tags;/

"wangyr: sometimes
set nowrap

"wangyr: for python
au BufEnter,BufRead *.py setlocal tabstop=4 expandtab shiftwidth=4 smartindent cinwords=if,elif,else,for,while,try,except,finally,def,class
"au BufRead,BufNewFile *.py,*pyw set shiftwidth=4
"au BufRead,BufNewFile *.py,*pyw set tabstop=4
"au BufRead,BufNewFile *.py,*.pyw set expandtab

"wangyr: for C-like programming, have automatic indentation and tab space of 2:
"binqin's set up autocmd FileType c,cpp,slang,cc,C,CC,h,hh,hpp setlocal tabstop=4 expandtab shiftwidth=4 smartindent noexpandtab ts=2
"When expandtab is set, hitting Tab in insert mode will produce the appropriate number of spaces.
"autocmd FileType c,cpp,slang,cc,C,CC,h,hh,hpp setlocal tabstop=2 expandtab shiftwidth=2 smartindent ts=2
autocmd FileType c,cpp,slang,cc,C,CC,h,hh,hpp setlocal tabstop=4 expandtab shiftwidth=4 smartindent ts=4

"wangyr: No sound/flash on errors
"should also see :help 'visualbell'
set noerrorbells
set novisualbell
set vb t_vb=
set tm=500

"wangyr: search without jump to prev/next identical character/word
nnoremap * *``
nnoremap # #``

"wangyr: continuous showing file name at the bottom of the file
set modeline
set ls=2

"wangyr: fixed the stupid python indentation issue where it shows # everytime start with a new line
inoremap # X<BS>#

"wangyr: search without jump to prev/next identical character/word, and without saving to the jumping list
"nnoremap * :keepjumps normal *``<cr>
"nnoremap # :keepjumps normal #``<cr>

" In an xterm the mouse should work quite well, thus enable it.
set mouse=a
set ttymouse=xterm2 "enable mouse mode for screen

" wangyr: to show sublime color
"colo xoria256

map <C-\> :tab split<CR>:exec("tag ".expand("<cword>"))<CR>
map <A-]> :vsp <CR>:exec("tag ".expand("<cword>"))<CR>

imap jj <Esc>

" wangyr: add a new line in normal mode
map <CR> o<Esc>
nnoremap <space> i<space><esc>

" scrollbind - to scroll multiple windows at the same time
" set scb
" set scb! for the other window
"
" for tmux

func! WordProcessorMode()
  setlocal formatoptions=1
  setlocal noexpandtab
  setlocal spell spelllang=en_us
  map j gj
  map k gk
  map <down> gj
  map <up> gk
  map $ g$
  map 0 g0
  set thesaurus+=/Users/wangyr/.vim/thesaurus/mthesaur.txt
  set complete+=s
  set formatprg=par
  setlocal wrap
  setlocal linebreak
endfu
com! WP call WordProcessorMode()

"Brandon's Additions"
set backupdir=~/vimtmp,.
set directory=~/vimtmp,.

" Make vim work with yaml better
autocmd FileType yaml setlocal ts=2 sts=2 sw=2 expandtab

"nerd tree"
autocmd vimenter * NERDTree
"close nerd tree if it's the only pane open
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
" Jump to the main window.
autocmd VimEnter * wincmd p

"enable copy and paste to clipboard
set clipboard=unnamed

set gfn=*
set gfn=Hack\ Nerd\ Font:h14

"have macvim use my color scheme"
let macvim_skip_colorscheme=1

" Vim-Airline Configuration
let g:airline#extensions#tabline#enabled = 1
let g:airline_powerline_fonts = 1 
let g:airline_theme='hybrid'
let g:hybrid_custom_term_colors = 1
let g:hybrid_reduced_contrast = 1 

" Yaml
au! BufNewFile,BufReadPost *.{yaml,yml} set filetype=yaml foldmethod=indent
autocmd FileType yaml setlocal ts=2 sts=2 sw=2 expandtab

" Have VI remember folds
augroup remember_folds
      autocmd!
        autocmd BufWinLeave * mkview
          autocmd BufWinEnter * silent! loadview
      augroup END
